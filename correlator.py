import os
import numpy as np
from typing import Optional, Tuple, Iterable, Dict, List
from concurrent.futures import ThreadPoolExecutor
from scipy import fft as sft
from scipy.signal import firwin, upfirdn
from utils import (get_fft_size, get_sample_count_of_file,
                   _NUMPY_DTYPE, get_bytes_per_sample)
from zc_sequence import create_zc


# DroneID sync (ZC) occupies ~9 MHz (600 carriers × 15 kHz). Decimating to
# the native LTE 10 MHz channel rate is sufficient for the correlator and
# keeps the fft_size = fs/15e3 relationship integer (1024).
_CORRELATOR_TARGET_RATE = 15.36e6

# Anti-alias LPF for the integer decimation step. 257 taps with a Kaiser
# β=8.5 window gives ~80 dB stopband attenuation, which is needed because
# an adjacent DroneID channel one slot away aliases back close to DC after
# folding around the decimated Nyquist.
_LPF_NTAPS = 257
_LPF_KAISER_BETA = 8.5

# Default chunk duration (seconds). Multiplied by sample_rate at runtime to
# obtain the number of samples per chunk.
_DEFAULT_CHUNK_DURATION_S = 0.680

# De-duplication window (decimated samples) around each above-threshold
# correlation score.
_PEAK_DEDUP_WINDOW = 10


def _precompute_filter_fft(filt: np.ndarray,
                           nfft: int) -> Tuple[np.ndarray, float]:
    """Pre-compute conj(FFT(zero_mean(filt))) and the filter's std-dev.

    These values are constant across all input chunks, so we compute them
    once and pass into normalized_xcorr_fast.
    """
    filt64 = filt.astype(np.complex128, copy=False)
    filt_zm = filt64 - filt64.mean()
    filter_var_sqrt = float(np.sqrt(np.var(filt_zm, ddof=1)))
    filt_zm_f32 = filt_zm.astype(np.complex64)
    H_conj = np.conj(sft.fft(filt_zm_f32, nfft, workers=-1))
    return H_conj, filter_var_sqrt


def _design_decimation_lpf(sample_rate: float,
                            dec_factor: int
                            ) -> Tuple[Optional[np.ndarray], int]:
    """Design a Kaiser-windowed anti-alias FIR for integer decimation.

    Cutoff is fixed at 5 MHz (= signal_bandwidth / 2) relative to absolute
    frequency; expressed as `10e6 / sample_rate` because firwin's cutoff
    is normalised to Nyquist. Returns (taps_f32, group_delay_samples). For
    dec_factor == 1 returns (None, 0).
    """
    if dec_factor <= 1:
        return None, 0
    taps = firwin(_LPF_NTAPS, 10e6 / sample_rate,
                  window=('kaiser', _LPF_KAISER_BETA)).astype(np.float32)
    group_delay = (_LPF_NTAPS - 1) // 2
    return taps, group_delay


def _make_ddc_filter(lpf_taps_real: np.ndarray,
                      freq_shift_hz: float,
                      sample_rate: float) -> np.ndarray:
    """Modulate real LPF taps so the polyphase output bandpass-selects at
    `freq_shift_hz` (= centre − target).

    Folding the explicit shift x_shifted[n] = x[n]·exp(+j·2π·f_shift·n/fs)
    into the polyphase convolution
        y[k] = Σ h[m]·x_shifted[k·D − m]
    yields h_mod[m] = h[m]·exp(−j·2π·f_shift·m/fs), with a leading
    exp(+j·2π·f_shift·k·D/fs) per output sample. That leading factor is a
    constant-frequency phase ramp in the decimated domain — magnitudes
    still match the explicit-shift path, but a standard DC-aligned ZC
    reference no longer correlates because the temporal phase pattern is
    rotated. The correlator must use a matching pre-shifted ZC reference
    (see _make_ddc_reference).
    """
    n = np.arange(len(lpf_taps_real), dtype=np.float64)
    mod = np.exp(-1j * 2.0 * np.pi * freq_shift_hz * n / sample_rate)
    return (lpf_taps_real * mod).astype(np.complex64)


def _make_ddc_reference(zc_taps: np.ndarray,
                         freq_shift_hz: float,
                         decimated_fs: float) -> np.ndarray:
    """Pre-rotate the ZC correlator reference to undo the DDC's residual
    per-sample phase ramp in the decimated domain.

    The DDC filter output at decimated sample k equals the DC-aligned
    decimated signal multiplied by exp(+j·2π·f_shift·k/fs_dec). To recover
    the correlation magnitude with the standard ZC, we conjugate this ramp
    into the reference:  ZC'[m] = ZC[m]·exp(−j·2π·f_shift·m/fs_dec).
    """
    m = np.arange(len(zc_taps), dtype=np.float64)
    rot = np.exp(-1j * 2.0 * np.pi * freq_shift_hz * m / decimated_fs)
    return (zc_taps * rot).astype(np.complex64)


def _resolve_decimation(sample_rate: float) -> Tuple[int, float]:
    """Pick an integer decimation factor that brings sample_rate close to
    the correlator target rate.

    Non-integer ratios fall back to dec_factor == 1 (no decimation).
    """
    dec_factor = max(1, int(round(sample_rate / _CORRELATOR_TARGET_RATE)))
    if dec_factor > 1 and sample_rate % dec_factor != 0:
        dec_factor = 1
    return dec_factor, sample_rate / dec_factor


def normalized_xcorr_fast(input_samples: np.ndarray,
                           filt: np.ndarray,
                           *,
                           filt_fft: Optional[np.ndarray] = None,
                           filter_var_sqrt: Optional[float] = None,
                           nfft: Optional[int] = None) -> np.ndarray:
    """FFT-based normalized cross-correlation (overlap-save, multithreaded).

    Optimizations vs the original single-large-FFT version:
      * scipy.fft with workers=-1 — uses every CPU core.
      * complex64 throughout — halves memory bandwidth, doubles SIMD lanes.
      * Overlap-save with a moderate FFT (4× filter length) so the working
        set fits in L2 cache.
      * Accepts a pre-computed filter FFT to avoid recomputing across chunks.

    Returns a complex vector of length len(input_samples) - len(filt).
    Use abs(result)**2 to get a score in [0, 1].
    """
    n_input = len(input_samples)
    n_filter = len(filt)
    n_scores = n_input - n_filter

    if n_scores <= 0:
        return np.array([], dtype=np.complex64)

    x = input_samples.astype(np.complex64, copy=False)

    if filt_fft is None or filter_var_sqrt is None:
        if nfft is None:
            nfft = sft.next_fast_len(4 * n_filter, real=False)
        H_conj, filter_var_sqrt = _precompute_filter_fft(filt, nfft)
    else:
        H_conj = filt_fft
        nfft = len(H_conj)

    stride = nfft - n_filter + 1

    # Overlap-save FFT correlation. Each block emits `stride` clean outputs;
    # the trailing (n_filter - 1) outputs are discarded as wrap-around noise.
    dot_products = np.empty(n_scores, dtype=np.complex64)
    with sft.set_workers(-1):
        for window_start in range(0, n_scores, stride):
            window_end = window_start + nfft
            if window_end > n_input:
                window = np.zeros(nfft, dtype=np.complex64)
                window[:n_input - window_start] = x[window_start:n_input]
            else:
                window = x[window_start:window_end]
            Y = sft.ifft(sft.fft(window, nfft) * H_conj)
            n_take = min(stride, n_scores - window_start)
            dot_products[window_start:window_start + n_take] = Y[:n_take]

    dot_products /= np.float32(n_filter)

    # Sliding-window variance via cumulative sums (float64 accumulator for
    # precision over millions of samples).
    abs_sq = x.real * x.real + x.imag * x.imag  # float32

    cs_abs = np.empty(n_input + 1, dtype=np.float64)
    cs_abs[0] = 0.0
    np.cumsum(abs_sq, out=cs_abs[1:])

    cs_re = np.empty(n_input + 1, dtype=np.float64)
    cs_re[0] = 0.0
    np.cumsum(x.real, out=cs_re[1:])

    cs_im = np.empty(n_input + 1, dtype=np.float64)
    cs_im[0] = 0.0
    np.cumsum(x.imag, out=cs_im[1:])

    w_energy = cs_abs[n_filter:n_filter + n_scores] - cs_abs[:n_scores]
    w_re = (cs_re[n_filter:n_filter + n_scores] - cs_re[:n_scores]) / n_filter
    w_im = (cs_im[n_filter:n_filter + n_scores] - cs_im[:n_scores]) / n_filter
    variance = (w_energy - n_filter * (w_re ** 2 + w_im ** 2)) / (n_filter - 1)

    denom = (np.sqrt(np.abs(variance)) * filter_var_sqrt).astype(np.float32)
    scores = dot_products / denom
    return scores


def _peaks_from_scores(abs_scores: np.ndarray,
                        correlation_threshold: float) -> List[int]:
    """De-duplicate above-threshold correlation samples into a peak list."""
    passing = np.where(abs_scores > correlation_threshold)[0]
    if len(passing) == 0:
        return []
    seen = set()
    half_win = _PEAK_DEDUP_WINDOW // 2
    n = len(abs_scores)
    for idx in passing:
        left_idx = idx - half_win
        right_idx = left_idx + _PEAK_DEDUP_WINDOW
        if left_idx < 0 or right_idx > n:
            continue
        window = abs_scores[left_idx:right_idx]
        seen.add(left_idx + int(np.argmax(window)))
    return sorted(seen)


def find_zc_indices_by_file(file_path: str, sample_rate: float,
                             frequency_offset: float,
                             correlation_threshold: float,
                             chunk_size: int,
                             sample_type: str = 'single',
                             chunk_duration_s: float =
                             _DEFAULT_CHUNK_DURATION_S) -> np.ndarray:
    """Single-frequency ZC search wrapper.

    `frequency_offset` is the Hz shift to apply to bring the target signal
    to DC (i.e. centre - target). All processing (decimation, chunking)
    lives in find_zc_indices_multi_freq; this wrapper just calls it with a
    one-element target list.

    `chunk_size` is kept for backward API compatibility but ignored — the
    new code path reads in chunks sized by `chunk_duration_s × sample_rate`.
    """
    del chunk_size  # unused in the new code path
    result = find_zc_indices_multi_freq(
        file_path, sample_rate,
        center_freq=0.0, target_freqs_hz=[-frequency_offset],
        correlation_threshold=correlation_threshold,
        chunk_duration_s=chunk_duration_s, sample_type=sample_type)
    return result[-frequency_offset]


def find_zc_indices_multi_freq(file_path: str, sample_rate: float,
                                center_freq: float,
                                target_freqs_hz: Iterable[float],
                                correlation_threshold: float,
                                chunk_duration_s: float =
                                _DEFAULT_CHUNK_DURATION_S,
                                sample_type: str = 'single'
                                ) -> Dict[float, np.ndarray]:
    """Scan a recording for ZC peaks at multiple target frequencies.

    All sample-count and bandwidth quantities are derived from the
    `sample_rate` argument, so the function works at any recording rate.

    The recording is read in chunks of `chunk_duration_s × sample_rate`
    samples (default 680 ms). For each chunk, every target frequency is
    processed independently:
      1. Frequency-shift to bring the target to DC.
      2. Anti-alias LPF (5 MHz cutoff) and integer polyphase decimation to
         ~15.36 MHz (or sample_rate if already at/below that rate).
      3. FFT-based ZC correlation.
      4. Threshold + dedup into a peak list.
    Detected peaks are returned in original-rate sample coordinates, with
    the LPF group delay compensated.

    Args:
        file_path: Path to the IQ recording.
        sample_rate: Recording sample rate in Hz.
        center_freq: SDR centre frequency in Hz.
        target_freqs_hz: Iterable of absolute target frequencies in Hz.
            Each must lie within [center_freq ± sample_rate/2].
        correlation_threshold: |score|^2 threshold in [0, 1].
        chunk_duration_s: Read chunk size in seconds (default 0.680).
        sample_type: I/Q sample data type ('int16', 'float32', …).

    Returns:
        dict mapping each target frequency (Hz) to a 1-D numpy array of
        sample offsets, expressed in original-rate samples.
    """
    target_freqs_hz = list(target_freqs_hz)
    dtype = _NUMPY_DTYPE[sample_type]
    bps = get_bytes_per_sample(sample_type)

    dec_factor, decimated_fs = _resolve_decimation(sample_rate)
    fft_size = get_fft_size(decimated_fs)
    correlator_taps = create_zc(fft_size, 4).astype(np.complex64)

    lpf_taps, lpf_delay = _design_decimation_lpf(sample_rate, dec_factor)

    # Pre-compute correlator filter FFT once — constant for every chunk
    # and every target frequency.
    nfft = sft.next_fast_len(4 * len(correlator_taps), real=False)
    H_conj, filter_var_sqrt = _precompute_filter_fft(correlator_taps, nfft)

    total_samples = get_sample_count_of_file(file_path, sample_type)
    chunk_samples = int(chunk_duration_s * sample_rate)
    # Align to dec_factor so chunk boundaries fall on decimation phase 0.
    if dec_factor > 1:
        chunk_samples = (chunk_samples // dec_factor) * dec_factor
    chunk_samples = max(chunk_samples, 1)

    print(f'There are {total_samples} samples in "{file_path}" '
          f'@ {sample_rate / 1e6:g} MHz')
    print(f'Chunk: {chunk_duration_s * 1000:g} ms = {chunk_samples} samples')
    if dec_factor > 1:
        print(f'Decimating {dec_factor}× for correlator: '
              f'{sample_rate / 1e6:g} → {decimated_fs / 1e6:g} MHz, '
              f'FFT {fft_size}, LPF {_LPF_NTAPS} taps Kaiser β={_LPF_KAISER_BETA}')
    else:
        print(f'No decimation (recording already at/below '
              f'{_CORRELATOR_TARGET_RATE / 1e6:g} MHz); FFT {fft_size}')
    print(f'Target frequencies ({len(target_freqs_hz)}):')
    for f in target_freqs_hz:
        print(f'  {f / 1e6:.3f} MHz  '
              f'(offset {(f - center_freq) / 1e6:+.3f} MHz from centre)')

    freq_to_peaks: Dict[float, List[int]] = {f: [] for f in target_freqs_hz}

    # Per-target precomputation. When decimating we use the DDC trick: the
    # frequency shift is baked into both the polyphase LPF taps and the
    # correlator's ZC reference, so the per-chunk inner loop is just one
    # upfirdn + one normalized_xcorr_fast (no full-rate shift, no per-sample
    # post-correction).
    ddc_filters: Dict[float, np.ndarray] = {}
    ddc_H_conj: Dict[float, np.ndarray] = {}
    ddc_var_sqrt: Dict[float, float] = {}
    if dec_factor > 1:
        for f in target_freqs_hz:
            f_shift = center_freq - f
            ddc_filters[f] = _make_ddc_filter(lpf_taps, f_shift, sample_rate)
            zc_ref = _make_ddc_reference(correlator_taps, f_shift,
                                          decimated_fs)
            ddc_H_conj[f], ddc_var_sqrt[f] = _precompute_filter_fft(
                zc_ref, nfft)

    # Thread budget: one thread per frequency, with the FFT workers split
    # across the threads so we don't oversubscribe the CPU.
    n_freqs = len(target_freqs_hz)
    cpu_count = os.cpu_count() or 1
    n_threads = min(n_freqs, cpu_count)
    fft_workers = max(1, cpu_count // max(n_threads, 1))
    if n_threads > 1:
        print(f'Per-chunk parallelism: {n_threads} threads × '
              f'{fft_workers} FFT workers (CPU count {cpu_count})')

    def _process_chunk_freq(freq: float,
                             chunk_complex: np.ndarray,
                             n_arange: Optional[np.ndarray]
                             ) -> Tuple[float, List[int]]:
        """Decimate + correlate one frequency, return (freq, decimated peaks)."""
        if dec_factor > 1:
            # DDC: one polyphase pass does shift + LPF + decimate. Pair
            # with the matching pre-shifted ZC reference (ddc_H_conj/var).
            decimated = upfirdn(ddc_filters[freq], chunk_complex,
                                 up=1, down=dec_factor)
            if decimated.dtype != np.complex64:
                decimated = decimated.astype(np.complex64, copy=False)
            H_conj_use = ddc_H_conj[freq]
            var_sqrt_use = ddc_var_sqrt[freq]
        else:
            # No-decimation path — explicit shift, original ZC reference.
            rot_const = 1j * np.pi * 2.0 * ((center_freq - freq) / sample_rate)
            rot = np.exp(rot_const * n_arange).astype(np.complex64)
            decimated = chunk_complex * rot
            H_conj_use = H_conj
            var_sqrt_use = filter_var_sqrt

        with sft.set_workers(fft_workers):
            scores = normalized_xcorr_fast(
                decimated, correlator_taps,
                filt_fft=H_conj_use,
                filter_var_sqrt=var_sqrt_use,
                nfft=nfft)
        abs_scores = np.abs(scores) ** 2
        return freq, _peaks_from_scores(abs_scores, correlation_threshold)

    sample_offset = 0
    chunk_num = 0

    with open(file_path, 'rb') as fh:
        pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None)
        try:
            while True:
                raw = np.frombuffer(fh.read(chunk_samples * 2 * bps),
                                     dtype=dtype)
                if len(raw) == 0:
                    break

                n_chunk_samples = len(raw) // 2
                if n_chunk_samples == 0:
                    break

                chunk_num += 1
                pct = min(100, round(sample_offset / total_samples * 100))
                print(f'  [{pct:3d}%] chunk {chunk_num}: samples '
                      f'{sample_offset} - {sample_offset + n_chunk_samples} of {total_samples}')

                # Interleaved I/Q -> complex64 (single float32 cast + view).
                raw_f32 = raw.astype(np.float32)
                chunk_complex = raw_f32.view(np.complex64)

                # Only needed for the no-decimation fallback path.
                n_arange = (None if dec_factor > 1 else
                            np.arange(sample_offset,
                                       sample_offset + n_chunk_samples,
                                       dtype=np.float64))

                if pool is not None:
                    futures = [pool.submit(_process_chunk_freq, f,
                                            chunk_complex, n_arange)
                               for f in target_freqs_hz]
                    chunk_results = [fut.result() for fut in futures]
                else:
                    chunk_results = [
                        _process_chunk_freq(f, chunk_complex, n_arange)
                        for f in target_freqs_hz]

                for freq, peaks in chunk_results:
                    for dec_peak in peaks:
                        abs_peak = (sample_offset + dec_peak * dec_factor
                                    - lpf_delay)
                        if 0 <= abs_peak < total_samples:
                            freq_to_peaks[freq].append(int(abs_peak))

                sample_offset += n_chunk_samples
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    print('  [100%] Search complete')

    result: Dict[float, np.ndarray] = {}
    for freq in target_freqs_hz:
        peaks = freq_to_peaks[freq]
        result[freq] = (np.unique(peaks) if peaks
                        else np.array([], dtype=int))
    return result
