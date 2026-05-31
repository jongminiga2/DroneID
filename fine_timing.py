"""Sub-sample timing and constant-phase fine-tuning.

Ports DroneDetection's Packet.find_zc_offset / Packet.find_zc_angle. After
coarse STO + CFO the OFDM windows are integer-sample-aligned, but a residual
fractional offset still puts a linear phase ramp across subcarriers. This
module removes that ramp using the first ZC symbol as a known reference.
"""

import numpy as np
from typing import Tuple

from utils import get_fft_size, get_frame_structure, with_sample_offset


def _zc_reference_freq(fft_size: int, ncarriers: int, root: int) -> np.ndarray:
    """Return the ZC reference on the active data carriers.

    Time-domain ZC: exp(-j*pi*root*n*(n+1)/(NCARRIERS+1)), n=0..NCARRIERS.
    The DC bin (index NCARRIERS//2) is dropped to match the OFDM mapping.
    """
    n = np.arange(ncarriers + 1, dtype=np.float64)
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / (ncarriers + 1))
    return np.delete(zc, ncarriers // 2)


def _extract_zc_symbol(burst: np.ndarray, fft_size: int,
                        cp_schedule: np.ndarray, zc_idx: int,
                        data_carrier_indices: np.ndarray) -> np.ndarray:
    """Extract ZC symbol's frequency-domain values on data carriers."""
    offset = int(cp_schedule[:zc_idx + 1].sum()) + fft_size * zc_idx
    sym = burst[offset : offset + fft_size]
    if len(sym) < fft_size:
        return np.zeros(len(data_carrier_indices), dtype=complex)
    spectrum = np.fft.fftshift(np.fft.fft(sym))
    return spectrum[data_carrier_indices]


def find_zc_offset(burst: np.ndarray, sample_rate: float, zc_idx: int,
                   zc_root: int, data_carrier_indices: np.ndarray,
                   search_range: float = 1.0, n_steps: int = 200,
                   legacy: bool = False) -> float:
    """Find the fractional sample shift that flattens ZC phase across subcarriers.

    Args:
        burst: Burst after integer STO and coarse CFO correction.
        sample_rate: Sample rate in Hz.
        zc_idx: 0-based burst index of the ZC symbol to use as reference.
        zc_root: ZC root index (600 or 147).
        data_carrier_indices: 600 active data carrier indices (after fftshift).
        search_range: Search ±search_range samples around 0.
        n_steps: Number of search steps in the range.
        legacy: True for Mavic Pro / Mavic 2 frame layout.

    Returns:
        Best fractional offset (samples) for with_sample_offset(burst, off).
    """
    fft_size = get_fft_size(sample_rate)
    structure = get_frame_structure(sample_rate, legacy=legacy)
    cp_schedule = structure['cp_schedule']

    ncarriers = len(data_carrier_indices)  # 600
    zc_ref = _zc_reference_freq(fft_size, ncarriers, zc_root)

    candidates = np.linspace(-search_range, search_range, n_steps)
    rms_scores = np.empty(n_steps)

    for i, offset in enumerate(candidates):
        shifted = with_sample_offset(burst, offset)
        zc_rx = _extract_zc_symbol(
            shifted, fft_size, cp_schedule, zc_idx, data_carrier_indices)

        # Prevent divide-by-zero on dead bins.
        zc_rx = np.where(zc_rx == 0, 1e-12 + 0j, zc_rx)

        adiff = np.angle(zc_ref / zc_rx)
        adiff = np.unwrap(adiff)
        rms_scores[i] = np.sqrt(np.mean((adiff - adiff.mean()) ** 2))

    return float(candidates[int(np.argmin(rms_scores))])


def find_zc_angle(burst: np.ndarray, sample_rate: float, zc_idx: int,
                  legacy: bool = False) -> float:
    """Return the constant phase rotation present at the ZC symbol's centre bin.

    DroneDetection uses `np.angle(symbol_f[NCARRIERS//2])`. With fftshift this
    corresponds to the centre (DC) bin of the OFDM spectrum.
    """
    fft_size = get_fft_size(sample_rate)
    structure = get_frame_structure(sample_rate, legacy=legacy)
    cp_schedule = structure['cp_schedule']

    offset = int(cp_schedule[:zc_idx + 1].sum()) + fft_size * zc_idx
    sym = burst[offset : offset + fft_size]
    spectrum = np.fft.fftshift(np.fft.fft(sym))
    return float(np.angle(spectrum[fft_size // 2]))
