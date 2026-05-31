import numpy as np
from typing import Iterable
from utils import (get_fft_size, get_frame_structure,
                   get_sample_count_of_file, read_complex)
from correlator import find_zc_indices_by_file


def extract_bursts_at_indices(input_path: str, sample_rate: float,
                                frequency_offset: float,
                                indices: Iterable[int],
                                padding: int,
                                sample_type: str = 'single',
                                legacy: bool = False) -> np.ndarray:
    """Read and frequency-correct bursts at the given ZC sample offsets.

    `indices` must be ZC-peak positions expressed in original-rate samples
    (i.e. what find_zc_indices_* returns). For each index the burst start
    is computed by backing off through the OFDM symbols preceding the
    first ZC plus `padding` head samples.

    Args:
        input_path: Path to IQ recording.
        sample_rate: Recording sample rate in Hz.
        frequency_offset: Frequency shift (Hz) to apply to each burst —
            usually centre_freq - target_freq for this set of indices.
        indices: ZC peak sample offsets (one entry per burst).
        padding: Extra samples to include before and after each burst.
        sample_type: I/Q sample data type.
        legacy: Use legacy 8-symbol frame layout.

    Returns:
        2-D complex array, shape (num_bursts, burst_sample_count).
    """
    structure = get_frame_structure(sample_rate, legacy=legacy)
    cp_schedule = structure['cp_schedule']
    num_symbols = structure['num_symbols']
    zc1_idx = structure['zc_symbol_indices'][0]
    fft_size = get_fft_size(sample_rate)

    num_file_samples = get_sample_count_of_file(input_path, sample_type)
    freq_offset_constant = 1j * np.pi * 2.0 * (frequency_offset / sample_rate)

    # Back off from the first ZC peak to the burst start: sum of CPs+FFTs of
    # all symbols before the first ZC, plus the first ZC's own CP.
    zc_seq_offset = int(cp_schedule[:zc1_idx + 1].sum()) + fft_size * zc1_idx
    burst_sample_count = ((padding * 2) + int(cp_schedule.sum())
                          + (fft_size * num_symbols))

    # Pre-compute frequency-offset correction vector (same for every burst).
    freq_offset_vec = np.exp(freq_offset_constant *
                              np.arange(1, burst_sample_count + 1))

    valid_starts = []
    for idx in indices:
        actual_start = int(idx) - padding - zc_seq_offset
        actual_end = actual_start + burst_sample_count

        if actual_start < 0:
            print(f"Warning: skipping burst at offset {idx} – start clipped")
            continue
        if actual_end > num_file_samples:
            print(f"Warning: skipping burst at offset {idx} – end clipped")
            continue
        valid_starts.append(actual_start)

    if not valid_starts:
        return np.zeros((0, burst_sample_count), dtype=complex)

    bursts = np.zeros((len(valid_starts), burst_sample_count), dtype=complex)
    for i, start in enumerate(valid_starts):
        burst = read_complex(input_path, start, burst_sample_count,
                              sample_type)
        bursts[i, :] = burst * freq_offset_vec
    return bursts


def extract_bursts_from_file(input_path: str, sample_rate: float,
                              frequency_offset: float,
                              correlation_threshold: float,
                              chunk_size: int, padding: int,
                              sample_type: str = 'single',
                              legacy: bool = False) -> np.ndarray:
    """Find ZC peaks then extract+frequency-correct each burst.

    Backwards-compatible wrapper: composes find_zc_indices_by_file with
    extract_bursts_at_indices. `chunk_size` is forwarded but the new
    correlator path uses a duration-based chunk size internally.

    Args:
        input_path: Path to IQ recording.
        sample_rate: Recording sample rate in Hz.
        frequency_offset: Frequency shift to apply before processing (Hz).
        correlation_threshold: ZC correlation threshold (0.0 – 1.0).
        chunk_size: Number of complex samples per read iteration.
        padding: Extra samples to include before and after each burst.
        sample_type: I/Q sample data type ('single', 'int16', etc.).
        legacy: Use legacy 8-symbol frame layout.

    Returns:
        2-D complex NumPy array, shape (num_bursts, burst_sample_count).
        Empty if no bursts found.
    """
    indices = find_zc_indices_by_file(
        input_path, sample_rate, frequency_offset,
        correlation_threshold, chunk_size, sample_type=sample_type)
    if len(indices) == 0:
        structure = get_frame_structure(sample_rate, legacy=legacy)
        fft_size = get_fft_size(sample_rate)
        burst_sample_count = ((padding * 2) + int(structure['cp_schedule'].sum())
                              + (fft_size * structure['num_symbols']))
        return np.zeros((0, burst_sample_count), dtype=complex)

    print(f'Extracting {len(indices)} burst(s)...')
    return extract_bursts_at_indices(
        input_path, sample_rate, frequency_offset, indices, padding,
        sample_type=sample_type, legacy=legacy)
