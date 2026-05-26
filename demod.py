import numpy as np


def quantize_qpsk(data_carriers: np.ndarray) -> np.ndarray:
    """Hard-decision QPSK demodulation.

    Replicates MATLAB's quantize_qpsk.m.

    Constellation mapping (same as openphy / LTE):
        +re, +im → 00
        +re, -im → 01
        -re, +im → 10
        -re, -im → 11

    Args:
        data_carriers: 1-D complex array of QPSK symbols.

    Returns:
        1-D int8 array of length 2 * len(data_carriers).
    """
    n = len(data_carriers)
    bits = np.zeros(n * 2, dtype=np.int8)
    bits[0::2] = (np.real(data_carriers) < 0).astype(np.int8)
    bits[1::2] = (np.imag(data_carriers) < 0).astype(np.int8)
    return bits
