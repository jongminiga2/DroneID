import numpy as np
from utils import (get_fft_size, get_cyclic_prefix_lengths,
                   get_sample_count_of_file, read_complex)
from correlator import find_zc_indices_by_file


def extract_bursts_from_file(input_path: str, sample_rate: float,
                              frequency_offset: float,
                              correlation_threshold: float,
                              chunk_size: int, padding: int,
                              sample_type: str = 'single') -> np.ndarray:
    """Extract all DroneID bursts from a raw IQ recording.

    Replicates MATLAB's extract_bursts_from_file.m.

    The returned matrix has one row per burst.  Each row contains the full
    burst (padding + 9 OFDM symbols + padding) with frequency offset already
    applied.

    Args:
        input_path: Path to IQ recording.
        sample_rate: Recording sample rate in Hz.
        frequency_offset: Frequency shift to apply before processing (Hz).
        correlation_threshold: ZC correlation threshold (0.0 – 1.0).
        chunk_size: Number of complex samples per read iteration.
        padding: Extra samples to include before and after each burst.
        sample_type: I/Q sample data type ('single', 'int16', etc.).

    Returns:
        2-D complex NumPy array, shape (num_bursts, burst_sample_count).
        Empty if no bursts found.
    """
    num_samples = get_sample_count_of_file(input_path, sample_type)

    fft_size = get_fft_size(sample_rate)
    long_cp_len, short_cp_len = get_cyclic_prefix_lengths(sample_rate)

    freq_offset_constant = 1j * np.pi * 2.0 * (frequency_offset / sample_rate)

    # The ZC correlator returns the index of the first sample *after* the ZC
    # symbol (i.e. the start of OFDM symbol 5's CP).  Back off to the true
    # start of the burst:  3 FFT windows + long_cp + 3 short_cps before that.
    zc_seq_offset = (fft_size * 3) + long_cp_len + (short_cp_len * 3)

    indices = find_zc_indices_by_file(
        input_path, sample_rate, frequency_offset,
        correlation_threshold, chunk_size, sample_type=sample_type)

    # 9 OFDM symbols: 2 long CPs, 7 short CPs
    burst_sample_count = (padding * 2) + (long_cp_len * 2) + (short_cp_len * 7) + (fft_size * 9)

    # Pre-compute frequency-offset correction vector (same for every burst)
    freq_offset_vec = np.exp(freq_offset_constant * np.arange(1, burst_sample_count + 1))

    valid_start_indices = []
    for idx in indices:
        actual_start = idx - padding - zc_seq_offset
        actual_end = actual_start + burst_sample_count

        if actual_start < 0:
            print(f"Warning: skipping burst at offset {idx} – start clipped")
            continue
        if actual_end > num_samples:
            print(f"Warning: skipping burst at offset {idx} – end clipped")
            continue
        valid_start_indices.append(actual_start)

    if not valid_start_indices:
        return np.zeros((0, burst_sample_count), dtype=complex)

    bursts = np.zeros((len(valid_start_indices), burst_sample_count), dtype=complex)
    print(f'Extracting {len(valid_start_indices)} burst(s)...')

    for i, start in enumerate(valid_start_indices):
        print(f'  Burst {i + 1} / {len(valid_start_indices)} (sample offset: {start})')
        burst = read_complex(input_path, start, burst_sample_count, sample_type)
        bursts[i, :] = burst * freq_offset_vec

    return bursts
