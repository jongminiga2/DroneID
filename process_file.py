#!/usr/bin/env python3
"""DJI DroneID burst processor – Python port of MATLAB's process_file.m.

Usage:
    python process_file.py --file <iq_file> [options]

    python process_file.py --file recording.bin \\
        --sample-type int16 --center-freq 2.4595e9 --target-freq 2.4595e9

The script:
  1. Searches the recording for ZC sequence correlations.
  2. Extracts each burst.
  3. Corrects integer and coarse frequency offsets.
  4. Demodulates all OFDM symbols.
  5. Descrambles and passes bits to the remove_turbo binary.
  6. Prints each decoded frame in hex and as JSON via parse_frame.py.
"""

import sys
import os
import argparse
import tempfile
import subprocess

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

# ---------------------------------------------------------------------------
# Make sure sibling modules are importable when called as a script
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import (get_fft_size, get_data_carrier_indices,
                   get_frame_structure, with_sample_offset)
from burst_extractor import extract_bursts_from_file
from timing import find_sto_cp
from ofdm import extract_ofdm_symbol_samples
from channel import calculate_channel
from demod import quantize_qpsk
from scrambler import generate_scrambler_seq
from fine_timing import find_zc_offset, find_zc_angle


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DJI DroneID IQ file processor")
    p.add_argument("--file", required=True, help="Path to IQ recording")
    p.add_argument("--sample-type", default="int16",
                   help="Sample type: single, float32, int16, int8, … (default: int16)")
    p.add_argument("--sample-rate", type=float, default=122.88e6,
                   help="Recording sample rate in Hz (default: 122.88 MHz)")
    p.add_argument("--center-freq", type=float, default=0.0,
                   help="SDR centre frequency in Hz (default: 0)")
    p.add_argument("--target-freq", type=float, default=0.0,
                   help="DroneID signal frequency in Hz (default: 0)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="ZC correlation threshold 0.0–1.0 (default: 0.7)")
    p.add_argument("--chunk-size", type=int, default=10_000_000,
                   help="Samples per read chunk (default: 10 000 000)")
    p.add_argument("--no-equalizer", action="store_true",
                   help="Disable frequency-domain equalizer")
    p.add_argument("--no-plots", action="store_true",
                   help="Disable matplotlib plots")
    p.add_argument("--legacy", action="store_true",
                   help="Decode legacy 8-symbol frame (Mavic Pro / Mavic 2)")
    p.add_argument("--fine-timing", action="store_true",
                   help="Enable sub-sample timing search (find_zc_offset)")
    p.add_argument("--fine-angle", action="store_true",
                   help="Enable ZC DC-bin phase correction (find_zc_angle)")
    p.add_argument("--ifo", action="store_true",
                   help="Enable upsampled IFO frequency-offset step "
                        "(off by default — the 10× round-trip can corrupt "
                        "timing at high sample rates; use only when "
                        "--target-freq is off by >7.5 kHz)")
    p.add_argument("--no-lpf", action="store_true",
                   help="Skip the low-pass filter step")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Locate sibling tools: remove_turbo and parse_frame.py
    # ------------------------------------------------------------------
    turbo_decoder_path = os.path.join(_HERE, "cpp", "remove_turbo")
    if sys.platform.startswith("win") and not os.path.isfile(turbo_decoder_path):
        turbo_decoder_path += ".exe"
    if not os.path.isfile(turbo_decoder_path):
        sys.exit(f"[ERROR] Cannot find remove_turbo at '{turbo_decoder_path}'. "
                 "Compile the C++ source first.")

    parse_frame_script = os.path.join(_HERE, "parse_frame.py")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    file_path = args.file
    sample_type = args.sample_type
    file_sample_rate = args.sample_rate
    center_freq = args.center_freq
    target_freq = args.target_freq
    file_freq_offset = center_freq - target_freq
    correlation_threshold = args.threshold
    chunk_size = args.chunk_size
    enable_equalizer = not args.no_equalizer
    enable_plots = not args.no_plots
    legacy = args.legacy

    # ------------------------------------------------------------------
    # Low-pass filter (matches MATLAB's fir1(50, 10e6/fs))
    # ------------------------------------------------------------------
    signal_bandwidth = 10e6
    filter_tap_count = 50
    # scipy firwin: cutoff normalised by Nyquist (same convention as MATLAB fir1)
    filter_taps = firwin(filter_tap_count + 1, signal_bandwidth / file_sample_rate)

    # ------------------------------------------------------------------
    # OFDM structure constants
    # ------------------------------------------------------------------
    fft_size = get_fft_size(file_sample_rate)
    structure = get_frame_structure(file_sample_rate, legacy=legacy)
    cp_schedule = structure['cp_schedule']
    num_symbols = structure['num_symbols']
    long_cp_len = structure['long_cp_len']
    short_cp_len = structure['short_cp_len']
    zc1_idx, zc2_idx = structure['zc_symbol_indices']
    data_symbol_indices = structure['data_symbol_indices']

    print(f"Frame layout: {'legacy (8 sym)' if legacy else 'modern (9 sym)'}, "
          f"ZC at 0-based indices {zc1_idx},{zc2_idx}")

    # ------------------------------------------------------------------
    # Burst extraction
    # ------------------------------------------------------------------
    bursts = extract_bursts_from_file(
        file_path, file_sample_rate, file_freq_offset,
        correlation_threshold, chunk_size,
        padding=filter_tap_count,
        sample_type=sample_type,
        legacy=legacy)

    if bursts.shape[0] == 0:
        sys.exit("[ERROR] No bursts found in the recording.")

    # ------------------------------------------------------------------
    # Pre-computed constants
    # ------------------------------------------------------------------
    data_carrier_indices = get_data_carrier_indices(file_sample_rate)

    # Initial state for the second LFSR in the scrambler (MATLAB: fliplr([...]))
    _x2_pre_flip = np.array([
        0, 0, 1,  0, 0, 1, 0,  0, 0, 1, 1,  0, 1, 0, 0,
        0, 1, 0, 1,  0, 1, 1, 0,  0, 1, 1, 1,  1, 0, 0, 0
    ], dtype=np.int32)
    scrambler_x2_init = _x2_pre_flip[::-1].copy()

    frames = []

    # ------------------------------------------------------------------
    # Per-burst processing
    # ------------------------------------------------------------------
    for burst_idx in range(bursts.shape[0]):
        burst = bursts[burst_idx].copy()

        if enable_plots:
            _plot_time_spectrum(burst, file_sample_rate, burst_idx)

        # --------------------------------------------------------------
        # Integer frequency offset (IFO) estimation via upsampled ZC
        # --------------------------------------------------------------
        # Index of the first data sample of the first ZC symbol,
        # accounting for the filter_tap_count padding at the burst head.
        ifo_offset = (int(cp_schedule[:zc1_idx + 1].sum())
                      + fft_size * zc1_idx + filter_tap_count)

        if args.ifo:
            interp_rate = 10
            burst = resample_poly(burst, interp_rate, 1)

            # Extract the upsampled ZC symbol data (no CP)
            zc_start = ifo_offset * interp_rate - 1
            zc_samples = burst[zc_start : zc_start + fft_size * interp_rate]

            fft_bins = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(zc_samples))) ** 2 + 1e-30)

            # Search ±15 bins around DC for the null (DC carrier)
            bin_count = 15
            center = len(fft_bins) // 2
            search = fft_bins.copy()
            search[: center - bin_count] = np.inf
            search[center + bin_count :] = np.inf
            center_offset = int(np.argmin(search))

            integer_offset_hz = (center - center_offset) * 15e3
            radians = 2 * np.pi * integer_offset_hz / (file_sample_rate * interp_rate)
            burst = burst * np.exp(1j * radians * np.arange(len(burst)))

            # Downsample back to original rate
            burst = resample_poly(burst, 1, interp_rate)

        # --------------------------------------------------------------
        # Low-pass filter
        # --------------------------------------------------------------
        if not args.no_lpf:
            burst = lfilter(filter_taps, [1.0], burst)

        # --------------------------------------------------------------
        # Symbol-timing offset (STO) correction via cyclic-prefix correlation
        # --------------------------------------------------------------
        interp_factor = 1   # set > 1 for sub-sample accuracy
        burst = resample_poly(burst, interp_factor, 1)
        true_start = find_sto_cp(burst, file_sample_rate * interp_factor,
                                  legacy=legacy)
        burst = resample_poly(burst[true_start:], 1, interp_factor)

        # --------------------------------------------------------------
        # Coarse carrier frequency offset (CFO) correction
        # using the cyclic prefix of the first ZC symbol
        # --------------------------------------------------------------
        # MATLAB 1-indexed value: last sample of the first ZC's CP
        zc_start_m = int(cp_schedule[:zc1_idx + 1].sum()) + fft_size * zc1_idx

        # Python slice equivalent of MATLAB burst(zc_start_m - short_cp_len : zc_start_m + fft_size - 1)
        # MATLAB arr(a:b) → Python arr[a-1 : b]
        cfo_start = zc_start_m - short_cp_len - 1
        cfo_end   = zc_start_m + fft_size - 1
        cfo_est_symbol = burst[cfo_start : cfo_end]

        cyclic_prefix = cfo_est_symbol[:short_cp_len]
        symbol_tail   = cfo_est_symbol[-short_cp_len:]

        # MATLAB: angle(dot(cp, tail)) / fft_size  — dot conjugates first arg
        offset_radians = np.angle(np.vdot(cyclic_prefix, symbol_tail)) / fft_size
        burst = burst * np.exp(-1j * offset_radians * np.arange(1, len(burst) + 1))

        # --------------------------------------------------------------
        # Sub-sample fine timing & constant-phase correction (DroneDetection-style)
        # --------------------------------------------------------------
        data_carrier_indices_pre = get_data_carrier_indices(file_sample_rate)
        # DroneDetection searches ±15 LTE samples; scale by Fs/15.36e6 here.
        scale = file_sample_rate / 15.36e6
        if args.fine_timing:
            sub_sample_offset = find_zc_offset(
                burst, file_sample_rate, zc_idx=zc1_idx, zc_root=600,
                data_carrier_indices=data_carrier_indices_pre,
                search_range=15.0 * scale, n_steps=600, legacy=legacy)
            if sub_sample_offset != 0.0:
                print(f"  Sub-sample offset: {sub_sample_offset:+.4f}")
                burst = with_sample_offset(burst, sub_sample_offset)

        if args.fine_angle:
            constant_phase = find_zc_angle(burst, file_sample_rate,
                                            zc_idx=zc1_idx, legacy=legacy)
            if constant_phase != 0.0:
                burst = burst * np.exp(-1j * constant_phase)

        # --------------------------------------------------------------
        # OFDM symbol extraction and channel estimation
        # --------------------------------------------------------------
        time_domain_syms, freq_domain_syms = extract_ofdm_symbol_samples(
            burst, file_sample_rate, legacy=legacy)

        # ZC root selection (4 → root 600, 6 → root 147) is the same for
        # modern and legacy; only the burst positions differ.
        channel1 = calculate_channel(freq_domain_syms[zc1_idx], file_sample_rate, 4)
        channel2 = calculate_channel(freq_domain_syms[zc2_idx], file_sample_rate, 6)

        channel1_data = channel1[data_carrier_indices]
        channel2_data = channel2[data_carrier_indices]

        channel1_phase = np.sum(np.angle(channel1_data)) / len(data_carrier_indices)
        channel2_phase = np.sum(np.angle(channel2_data)) / len(data_carrier_indices)
        # channel_phase_adj computed but not applied (matches MATLAB behaviour)
        _channel_phase_adj = (channel1_phase - channel2_phase) / 2  # noqa: F841

        channel = channel1_data  # MATLAB behaviour: first ZC only

        if enable_plots:
            _plot_constellations(freq_domain_syms, data_carrier_indices, channel,
                                 enable_equalizer, burst_idx + 1, _HERE,
                                 num_symbols=num_symbols)

        # --------------------------------------------------------------
        # QPSK demodulation + descramble + turbo, trying 4 phase rotations
        # (DroneDetection-style — CRC selects the correct alignment)
        # --------------------------------------------------------------
        second_scrambler = generate_scrambler_seq(7200, scrambler_x2_init)
        bits_tmp = os.path.join(tempfile.gettempdir(), "droneid_bits")

        burst_frame = ""
        last_err = ""
        for phase_idx in range(4):
            phase_rot = np.exp(1j * np.pi / 2 * phase_idx)

            bits_mat = np.zeros((num_symbols, 1200), dtype=np.int8)
            for sym_idx in range(num_symbols):
                dc = freq_domain_syms[sym_idx, data_carrier_indices]
                if enable_equalizer:
                    dc = dc * channel
                bits_mat[sym_idx] = quantize_qpsk(dc * phase_rot)

            bits_sel = bits_mat[data_symbol_indices, :]
            bits = bits_sel.flatten().astype(np.int32)
            bits = np.bitwise_xor(bits, second_scrambler).astype(np.int8)

            bits.tofile(bits_tmp)
            result = subprocess.run(
                [turbo_decoder_path, bits_tmp],
                capture_output=True, text=True)

            if result.returncode == 0:
                print(f"phase {phase_idx*90}°: {int(np.sum(bits == 1))} ones / "
                      f"{int(np.sum(bits == 0))} zeros → CRC OK")
                burst_frame = result.stdout
                break
            else:
                last_err = result.stderr.strip()
                print(f"phase {phase_idx*90}°: {int(np.sum(bits == 1))} ones / "
                      f"{int(np.sum(bits == 0))} zeros → {last_err}")

        if not burst_frame:
            print(f"Warning: all 4 phase rotations failed for burst {burst_idx + 1}")
        frames.append(burst_frame)

    # ------------------------------------------------------------------
    # Print decoded frames and parse to JSON
    # ------------------------------------------------------------------
    for idx, frame in enumerate(frames):
        print(f"FRAME: {frame}", end="")
        if frame.strip():
            ret = subprocess.run(
                [sys.executable, parse_frame_script, frame.strip()],
                capture_output=True, text=True)
            if ret.returncode == 0:
                print(ret.stdout)
            else:
                print(f"Warning: parse_frame.py failed (exit {ret.returncode})")


# ---------------------------------------------------------------------------
# Optional plotting helpers (imported lazily so matplotlib is not required)
# ---------------------------------------------------------------------------

def _plot_time_spectrum(burst: np.ndarray, fs: float, idx: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, num=43)
    axes[0].plot(10 * np.log10(np.abs(burst) ** 2 + 1e-30))
    axes[0].set_title("Time domain |x|² 10log10 (original)")

    fft_bins = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(burst))) ** 2 + 1e-30)
    x_axis = np.linspace(-fs / 2, fs / 2, len(burst))
    axes[1].plot(x_axis, fft_bins)
    axes[1].set_title("Frequency spectrum")
    axes[1].grid(True)
    plt.tight_layout()
    plt.pause(0.01)


def _plot_constellations(freq_syms: np.ndarray, dc_idx: np.ndarray,
                          channel: np.ndarray, equalize: bool,
                          burst_num: int, save_dir: str,
                          num_symbols: int = 9) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Grid: 3x3 for modern (9), 2x4 for legacy (8)
    if num_symbols <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, num=1, figsize=(cols * 3, rows * 3))
    for sym_idx, ax in enumerate(axes.flat):
        if sym_idx >= num_symbols:
            ax.axis('off')
            continue
        dc = freq_syms[sym_idx, dc_idx]
        if equalize:
            dc = dc * channel
        ax.plot(dc.real, dc.imag, 'o', markersize=1)
        ax.set_title(f"Symbol {sym_idx + 1} IQ")
    plt.tight_layout()
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, f"ofdm_symbol_{burst_num}.png"))
    plt.pause(0.01)


if __name__ == "__main__":
    main()
