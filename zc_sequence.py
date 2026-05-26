import numpy as np
from utils import get_data_carrier_indices


def create_zc(fft_size: int, symbol_index: int) -> np.ndarray:
    """Create a time-domain ZC sequence mapped onto an OFDM symbol.

    Replicates MATLAB's create_zc.m.  The result can be cross-correlated
    against a recording to detect OFDM symbol 4 (root 600) or 6 (root 147).

    Args:
        fft_size: OFDM FFT window size (must be a power of 2).
        symbol_index: 4 or 6.

    Returns:
        Complex time-domain samples of length fft_size (no cyclic prefix).
    """
    assert symbol_index in (4, 6), "symbol_index must be 4 or 6"
    assert fft_size > 0 and (fft_size & (fft_size - 1)) == 0, \
        "fft_size must be a power of 2"

    root = 600 if symbol_index == 4 else 147

    # 601-point ZC sequence: exp(-j*pi*root*n*(n+1)/601) for n=0..600
    n = np.arange(601, dtype=np.float64)
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / 601)

    # Remove element at index 300 (MATLAB index 301) — the DC carrier
    zc = np.delete(zc, 300)  # now 600 elements

    # Map ZC values onto data carriers in the frequency domain
    samples_freq = np.zeros(fft_size, dtype=complex)
    data_carrier_indices = get_data_carrier_indices(fft_size * 15e3)
    samples_freq[data_carrier_indices] = zc

    # Convert to time domain (flip spectrum before IFFT, matching MATLAB)
    return np.fft.ifft(np.fft.fftshift(samples_freq))
