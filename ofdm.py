import numpy as np
from typing import Tuple
from utils import get_fft_size, get_frame_structure


def extract_ofdm_symbol_samples(samples: np.ndarray,
                                 sample_rate: float,
                                 legacy: bool = False,
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Strip cyclic prefixes and return time- and frequency-domain symbols.

    Replicates MATLAB's extract_ofdm_symbol_samples.m.

    Args:
        samples: Time-domain burst starting exactly at the first CP sample.
        sample_rate: Sample rate in Hz.
        legacy: If True, use 8-symbol Mavic Pro / Mavic 2 frame layout.

    Returns:
        time_domain:  (num_symbols, fft_size) complex array of time-domain symbol data.
        freq_domain:  (num_symbols, fft_size) complex array after fftshift(fft(...)).
    """
    fft_size = get_fft_size(sample_rate)
    structure = get_frame_structure(sample_rate, legacy=legacy)
    cp_lengths = structure['cp_schedule']
    num_symbols = structure['num_symbols']
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
