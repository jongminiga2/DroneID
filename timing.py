import numpy as np
from utils import get_fft_size, get_cyclic_prefix_lengths


def find_sto_cp(samples: np.ndarray, sample_rate: float) -> int:
    """Estimate the burst start offset using cyclic-prefix correlation.

    Replicates MATLAB's find_sto_cp.m.

    Immune to frequency-offset-induced time-shift that affects ZC correlation.
    Provide an upsampled burst for sub-sample accuracy (though interp_factor=1
    is used in the main script so the burst is at the original sample rate).

    Args:
        samples: Complex IQ burst samples (first sample near but before CP1).
        sample_rate: Sample rate of the provided samples in Hz.

    Returns:
        0-based index of the first sample of CP1.
    """
    long_cp_len, short_cp_len = get_cyclic_prefix_lengths(sample_rate)
    fft_size = get_fft_size(sample_rate)

    cp_schedule = np.array([
        long_cp_len,
        short_cp_len, short_cp_len, short_cp_len,
        short_cp_len, short_cp_len, short_cp_len,
        short_cp_len,
        long_cp_len,
    ])
    num_symbols = len(cp_schedule)
    full_burst_len = int(cp_schedule.sum()) + fft_size * num_symbols
    num_tests = len(samples) - full_burst_len

    scores_cp_sto = np.zeros(num_tests)

    for test_idx in range(num_tests):
        offset = test_idx
        scores = np.zeros(num_symbols)
        for sym_idx, cp_len in enumerate(cp_schedule):
            window = samples[offset : offset + fft_size + cp_len]
            left = window[:cp_len]
            right = window[-cp_len:]
            # Magnitude of zero-lag cross-correlation between CP and symbol tail
            scores[sym_idx] = abs(np.dot(left, right.conj()))
            offset += fft_size + cp_len
        # Skip first symbol (may be absent on some drones); average the rest
        scores_cp_sto[test_idx] = scores[1:].sum() / (num_symbols - 1)

    return int(np.argmax(scores_cp_sto))
