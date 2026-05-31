import numpy as np
import os
from typing import Tuple, Dict, Any


def get_fft_size(sample_rate: float) -> int:
    return int(sample_rate / 15e3)


def get_cyclic_prefix_lengths(sample_rate: float) -> Tuple[int, int]:
    long_cp_len = round(1 / 192000 * sample_rate)
    short_cp_len = round(0.0000046875 * sample_rate)
    return long_cp_len, short_cp_len


def get_data_carrier_indices(sample_rate: float) -> np.ndarray:
    fft_size = get_fft_size(sample_rate)
    data_carrier_count = 600
    dc_idx = fft_size // 2  # 0-indexed DC bin (= MATLAB's fft_size/2 + 1, minus 1)
    mapping = np.zeros(fft_size, dtype=np.int32)
    mapping[dc_idx - data_carrier_count // 2 : dc_idx] = 1
    mapping[dc_idx + 1 : dc_idx + data_carrier_count // 2 + 1] = 1
    return np.where(mapping == 1)[0]


def get_frame_structure(sample_rate: float, legacy: bool = False) -> Dict[str, Any]:
    """OFDM frame layout for DroneID.

    Modern (post-Mavic 2): 9 OFDM symbols, ZC at 0-based indices [3, 5],
    symbol 0 carries a known Gold sequence and is excluded from payload.

    Legacy (Mavic Pro / Mavic 2): 8 OFDM symbols (symbol 0 missing in tx),
    ZC at 0-based indices [2, 4]. All non-ZC symbols carry payload data.
    """
    long_cp_len, short_cp_len = get_cyclic_prefix_lengths(sample_rate)

    if legacy:
        cp_schedule = np.array([
            long_cp_len,
            short_cp_len, short_cp_len, short_cp_len,
            short_cp_len, short_cp_len, short_cp_len,
            long_cp_len,
        ])
        zc_symbol_indices = (2, 4)
        data_symbol_indices = [0, 1, 3, 5, 6, 7]
    else:
        cp_schedule = np.array([
            long_cp_len,
            short_cp_len, short_cp_len, short_cp_len,
            short_cp_len, short_cp_len, short_cp_len,
            short_cp_len,
            long_cp_len,
        ])
        zc_symbol_indices = (3, 5)
        data_symbol_indices = [1, 2, 4, 6, 7, 8]

    return {
        'cp_schedule': cp_schedule,
        'num_symbols': len(cp_schedule),
        'zc_symbol_indices': zc_symbol_indices,
        'data_symbol_indices': data_symbol_indices,
        'long_cp_len': long_cp_len,
        'short_cp_len': short_cp_len,
        'legacy': legacy,
    }


_BYTES_PER_SAMPLE = {
    'single': 4, 'float32': 4,
    'double': 8, 'float64': 8,
    'int16': 2, 'int8': 1,
    'uint8': 1, 'int32': 4,
    'uint16': 2, 'uint32': 4,
}

_NUMPY_DTYPE = {
    'single': np.float32, 'float32': np.float32,
    'double': np.float64, 'float64': np.float64,
    'int16': np.int16, 'int8': np.int8,
    'uint8': np.uint8, 'int32': np.int32,
    'uint16': np.uint16, 'uint32': np.uint32,
}


def get_bytes_per_sample(sample_type: str) -> int:
    bps = _BYTES_PER_SAMPLE.get(sample_type)
    if bps is None:
        raise ValueError(f"Unknown sample type: {sample_type!r}")
    return bps


def get_sample_count_of_file(file_path: str, sample_type: str) -> int:
    bps = get_bytes_per_sample(sample_type)
    return os.path.getsize(file_path) // bps // 2


def with_sample_offset(data: np.ndarray, offset: float) -> np.ndarray:
    """Shift complex samples by a fractional offset via linear interpolation.

    Ports DroneDetection's helpers.with_sample_offset. Equivalent to taking
    data at indices [offset, offset+1, ..., offset+len-1]; values outside the
    original range are extrapolated by numpy.interp's edge-clamp behaviour.
    """
    n = len(data)
    x_new = np.arange(offset, offset + n)
    x_old = np.arange(n)
    if np.iscomplexobj(data):
        return (np.interp(x_new, x_old, data.real)
                + 1j * np.interp(x_new, x_old, data.imag))
    return np.interp(x_new, x_old, data)


def read_complex(file_path: str, sample_offset: int, sample_count: int,
                 sample_type: str) -> np.ndarray:
    dtype = _NUMPY_DTYPE.get(sample_type)
    if dtype is None:
        raise ValueError(f"Unknown sample type: {sample_type!r}")
    bps = get_bytes_per_sample(sample_type)
    with open(file_path, 'rb') as f:
        f.seek(sample_offset * bps * 2)
        raw = np.frombuffer(f.read(sample_count * 2 * bps), dtype=dtype)
    return raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
