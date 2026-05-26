import numpy as np
from utils import get_fft_size
from zc_sequence import create_zc


def calculate_channel(zc_seq: np.ndarray, sample_rate: float,
                       symbol_idx: int) -> np.ndarray:
    """Estimate channel taps from a received ZC symbol.

    Replicates MATLAB's calculate_channel.m.

    Divides a golden-reference frequency-domain ZC sequence by the received
    version to obtain zero-forcing equalizer coefficients.

    Args:
        zc_seq: Frequency-domain samples of OFDM symbol 4 or 6
                (full fft_size bins, from fftshift(fft(...))).
        sample_rate: Sample rate in Hz.
        symbol_idx: 4 or 6.

    Returns:
        Complex array of length fft_size containing equalizer taps.
    """
    assert symbol_idx in (4, 6), "symbol_idx must be 4 or 6"
    fft_size = get_fft_size(sample_rate)
    gold_time = create_zc(fft_size, symbol_idx)
    gold_freq = np.fft.fftshift(np.fft.fft(gold_time))
    return gold_freq / zc_seq
