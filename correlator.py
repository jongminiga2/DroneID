import numpy as np
from typing import Optional
from utils import get_fft_size, get_sample_count_of_file, _NUMPY_DTYPE
from zc_sequence import create_zc


def normalized_xcorr_fast(input_samples: np.ndarray,
                           filt: np.ndarray) -> np.ndarray:
    """FFT-based normalized cross-correlation.

    Replicates MATLAB's normalized_xcorr_fast.m.

    Returns a complex vector of length len(input_samples) - len(filt).
    Correlation peaks point to the *beginning* of the filter sequence.
    Use abs(result)**2 to get a score in the range [0, 1].
    """
    n_input = len(input_samples)
    n_filter = len(filt)
    n_scores = n_input - n_filter

    if n_scores <= 0:
        return np.array([], dtype=complex)

    # Zero-mean the filter
    filt_zm = filt - np.mean(filt)
    filter_var_sqrt = np.sqrt(np.var(filt_zm, ddof=1))  # MATLAB var uses N-1

    # FFT-based cross-correlation
    nfft = 1 << int(np.ceil(np.log2(n_input + n_filter - 1)))
    corr_full = np.fft.ifft(
        np.fft.fft(input_samples, nfft) * np.conj(np.fft.fft(filt_zm, nfft))
    )
    dot_products = corr_full[:n_scores] / n_filter

    # Sliding-window variance via cumulative sums
    x = input_samples
    abs_sq = np.real(x) ** 2 + np.imag(x) ** 2
    cs_abs = np.concatenate([[0.0], np.cumsum(abs_sq)])
    cs_re = np.concatenate([[0.0], np.cumsum(np.real(x))])
    cs_im = np.concatenate([[0.0], np.cumsum(np.imag(x))])

    w_energy = cs_abs[n_filter : n_filter + n_scores] - cs_abs[:n_scores]
    w_re = (cs_re[n_filter : n_filter + n_scores] - cs_re[:n_scores]) / n_filter
    w_im = (cs_im[n_filter : n_filter + n_scores] - cs_im[:n_scores]) / n_filter
    variance = (w_energy - n_filter * (w_re ** 2 + w_im ** 2)) / (n_filter - 1)

    scores = dot_products / (np.sqrt(np.abs(variance)) * filter_var_sqrt)
    return scores


def find_zc_indices_by_file(file_path: str, sample_rate: float,
                             frequency_offset: float,
                             correlation_threshold: float,
                             chunk_size: int,
                             sample_type: str = 'single') -> np.ndarray:
    """Find sample offsets of the first ZC sequence throughout a recording.

    Replicates MATLAB's find_zc_indices_by_file.m.

    Args:
        file_path: Path to the IQ recording file.
        sample_rate: Recording sample rate in Hz.
        frequency_offset: Frequency shift to apply (Hz) to centre the signal.
        correlation_threshold: Minimum |score|^2 threshold (0.0 – 1.0).
        chunk_size: Samples to read per iteration (complex samples).
        sample_type: Data type of each I/Q value ('single', 'int16', etc.).

    Returns:
        1-D array of 0-indexed sample offsets where the ZC sequence was found.
    """
    fft_size = get_fft_size(sample_rate)
    dtype = _NUMPY_DTYPE[sample_type]
    from utils import get_bytes_per_sample
    bps = get_bytes_per_sample(sample_type)

    freq_offset_constant = 1j * np.pi * 2.0 * (frequency_offset / sample_rate)

    # Reference ZC sequence (OFDM symbol 4, root 600) used as correlator taps
    correlator_taps = create_zc(fft_size, 4)

    total_samples = get_sample_count_of_file(file_path, sample_type)
    print(f'There are {total_samples} samples in "{file_path}"')

    zc_scores_list = []
    leftover_samples = np.array([], dtype=complex)
    sample_offset = 0
    chunk_num = 0

    with open(file_path, 'rb') as fh:
        while True:
            raw = np.frombuffer(fh.read(chunk_size * 2 * bps), dtype=dtype)
            if len(raw) == 0:
                break

            new_samples = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)

            if len(new_samples) + len(leftover_samples) <= len(correlator_taps):
                break

            chunk_num += 1
            pct = min(100, round(sample_offset / total_samples * 100))
            print(f'  [{pct:3d}%] chunk {chunk_num}: samples '
                  f'{sample_offset} - {sample_offset + len(new_samples)} of {total_samples}')

            # Frequency-shift new samples (leftover was already shifted)
            rot = np.exp(freq_offset_constant *
                         np.arange(sample_offset, sample_offset + len(new_samples)))
            new_samples = new_samples * rot

            # Prepend leftover from previous chunk for seamless correlation
            samples = np.concatenate([leftover_samples, new_samples])
            leftover_samples = samples[-len(correlator_taps):]

            scores = normalized_xcorr_fast(samples, correlator_taps)
            zc_scores_list.append(scores)

            # Replicate MATLAB's sample_offset increment (includes leftover length)
            sample_offset += len(samples)

    print('  [100%] Search complete')

    if not zc_scores_list:
        return np.array([], dtype=int)

    zc_scores = np.concatenate(zc_scores_list)
    abs_scores = np.abs(zc_scores) ** 2

    # Find all indices above threshold
    passing = np.where(abs_scores > correlation_threshold)[0]
    if len(passing) == 0:
        return np.array([], dtype=int)

    # Keep only the peak within a small search window to eliminate duplicates
    search_window = 10
    true_peaks = []
    for idx in passing:
        left_idx = idx - search_window // 2
        right_idx = left_idx + search_window
        if left_idx < 0 or right_idx > len(abs_scores):
            continue
        window = abs_scores[left_idx:right_idx]
        peak_in_window = left_idx + int(np.argmax(window))
        true_peaks.append(peak_in_window)

    return np.unique(true_peaks)
