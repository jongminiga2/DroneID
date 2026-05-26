import numpy as np


def generate_scrambler_seq(num_bits: int, x2_init: np.ndarray) -> np.ndarray:
    """Generate an LTE Gold-sequence scrambler bit stream.

    Replicates MATLAB's generate_scrambler_seq.m.

    References:
        3GPP TS 36.211 §7.2

    Args:
        num_bits: Number of output bits to generate.
        x2_init: 31-bit initial state for the second LFSR (x2).
                 Must be a 1-D array of 0/1 values of length 31.

    Returns:
        1-D int32 array of length num_bits containing 0/1 values.
    """
    assert len(x2_init) == 31, "x2_init must be 31 bits"

    Nc = 1600
    total = Nc + num_bits + 31

    x1 = np.zeros(total, dtype=np.int32)
    x2 = np.zeros(total, dtype=np.int32)

    # Fixed initial state for x1 (3GPP 36.211 §7.2)
    x1[0] = 1

    x2[:31] = x2_init.astype(np.int32)

    # Generate m-sequences
    for n in range(Nc + num_bits):
        x1[n + 31] = (x1[n + 3] + x1[n]) % 2
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) % 2

    # Gold sequence output
    c = (x1[Nc : Nc + num_bits] + x2[Nc : Nc + num_bits]) % 2
    return c.astype(np.int32)
