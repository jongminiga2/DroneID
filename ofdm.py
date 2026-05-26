import numpy as np
from typing import Tuple
from utils import get_fft_size, get_cyclic_prefix_lengths


def extract_ofdm_symbol_samples(samples: np.ndarray,
                                 sample_rate: float
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Strip cyclic prefixes and return time- and frequency-domain symbols.

    Replicates MATLAB's extract_ofdm_symbol_samples.m.

    Args:
        samples: Time-domain burst starting exactly at the first CP sample.
        sample_rate: Sample rate in Hz.

    Returns:
        time_domain:  (9, fft_size) complex array of time-domain symbol data.
        freq_domain:  (9, fft_size) complex array after fftshift(fft(...)).
    """
    fft_size = get_fft_size(sample_rate)
    long_cp_len, short_cp_len = get_cyclic_prefix_lengths(sample_rate)

    cp_lengths = np.array([
        long_cp_len,
        short_cp_len, short_cp_len, short_cp_len,
        short_cp_len, short_cp_len, short_cp_len,
        short_cp_len,
        long_cp_len,
    ])

    num_symbols = len(cp_lengths)
    time_domain = np.zeros((num_symbols, fft_size), dtype=complex)
    freq_domain = np.zeros((num_symbols, fft_size), dtype=complex)

    offset = 0
    for idx, cp_len in enumerate(cp_lengths):
        # Extract CP + data, then discard the CP
        symbol = samples[offset : offset + fft_size + cp_len]
        symbol = symbol[cp_len:]               # fft_size samples
        time_domain[idx] = symbol
        freq_domain[idx] = np.fft.fftshift(np.fft.fft(symbol))
        offset += fft_size + cp_len

    return time_domain, freq_domain
