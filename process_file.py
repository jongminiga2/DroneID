#!/usr/bin/env python3
"""DJI DroneID burst processor – Python port of MATLAB's process_file.m.

Usage:
    python process_file.py --file <iq_file> [options]

    # Single-frequency override (legacy behaviour)
    python process_file.py --file recording.bin \\
        --sample-type int16 --center-freq 2.4595e9 --target-freq 2.4595e9

    # Multi-frequency scan (default — uses marker_freqs.json)
    python process_file.py --file recording.bin \\
        --sample-type int16 --center-freq 2.45e9

The script:
  1. Determines the set of target frequencies (single via --target-freq, or
     every entry in marker_freqs.json that falls within ±sample_rate/2 of
     the centre frequency).
  2. Scans the recording for ZC correlations at each target.
  3. Extracts each burst.
  4. Corrects integer and coarse frequency offsets.
  5. Demodulates all OFDM symbols.
  6. Descrambles and passes bits to the remove_turbo binary.
  7. Prints each decoded frame in hex and as JSON via parse_frame.py.
"""

import sys
import os
import json
import argparse
import tempfile
import subprocess
from typing import List, Optional

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
from burst_extractor import extract_bursts_at_indices
from correlator import find_zc_indices_multi_freq
from timing import find_sto_cp
from ofdm import extract_ofdm_symbol_samples
from channel import calculate_channel
from demod import quantize_qpsk
from scrambler import generate_scrambler_seq
from fine_timing import find_zc_offset, find_zc_angle


_MARKER_FREQS_JSON = os.path.join(_HERE, "marker_freqs.json")


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
                   help="DroneID signal frequency in Hz. If 0 (default), "
                        "scan all marker_freqs.json entries within "
                        "centre_freq ± sample_rate/2.")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="ZC correlation threshold 0.0–1.0 (default: 0.7)")
    p.add_argument("--chunk-duration", type=float, default=0.680,
                   help="Correlator chunk size in seconds (default: 0.680). "
                        "Multiplied by --sample-rate to size each read.")
    p.add_argument("--marker-freqs",
                   default=_MARKER_FREQS_JSON,
                   help="Path to marker frequencies JSON "
                        "(default: marker_freqs.json next to this script)")
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
# Marker frequency loading
# ---------------------------------------------------------------------------

def load_in_band_marker_freqs(marker_path: str, center_freq: float,
                               sample_rate: float) -> List[float]:
    """Return the frequencies (Hz) from marker_freqs.json that lie within
    [center_freq - sample_rate/2, center_freq + sample_rate/2].
    """
    if not os.path.isfile(marker_path):
        return []
    with open(marker_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    freqs_hz = [float(f) * 1e6 for f in data.get("frequencies_mhz", [])]
    half_bw = sample_rate / 2.0
    return [f for f in freqs_hz
            if (center_freq - half_bw) <= f <= (center_freq + half_bw)]


# ---------------------------------------------------------------------------
# Per-burst decode (factored out so it can run for multiple frequencies)
# ---------------------------------------------------------------------------

def decode_burst(burst: np.ndarray, *,
                 sample_rate: float,
                 args: argparse.Namespace,
                 filter_taps: np.ndarray,
                 filter_tap_count: int,
                 structure: dict,
                 fft_size: int,
                 data_carrier_indices: np.ndarray,
                 scrambler_x2_init: np.ndarray,
                 turbo_decoder_path: str,
                 burst_idx: int,
                 burst_label: str) -> str:
    """Run the full per-burst decode chain. Returns the decoded frame hex
    string from remove_turbo, or "" on failure.
    """
    cp_schedule = structure['cp_schedule']
    num_symbols = structure['num_symbols']
    short_cp_len = structure['short_cp_len']
    zc1_idx, zc2_idx = structure['zc_symbol_indices']
    data_symbol_indices = structure['data_symbol_indices']
    enable_equalizer = not args.no_equalizer
    enable_plots = not args.no_plots
    legacy = args.legacy

    if enable_plots:
        _plot_time_spectrum(burst, sample_rate, burst_idx, burst_label)

    # Integer frequency offset (IFO) estimation via upsampled ZC ------------
    ifo_offset = (int(cp_schedule[:zc1_idx + 1].sum())
                  + fft_size * zc1_idx + filter_tap_count)

    if args.ifo:
        interp_rate = 10
        burst = resample_poly(burst, interp_rate, 1)
        zc_start = ifo_offset * interp_rate - 1
        zc_samples = burst[zc_start: zc_start + fft_size * interp_rate]
        fft_bins = 10 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(zc_samples))) ** 2 + 1e-30)
        bin_count = 15
        center = len(fft_bins) // 2
        search = fft_bins.copy()
        search[: center - bin_count] = np.inf
        search[center + bin_count:] = np.inf
        center_offset = int(np.argmin(search))
        integer_offset_hz = (center - center_offset) * 15e3
        radians = 2 * np.pi * integer_offset_hz / (sample_rate * interp_rate)
        burst = burst * np.exp(1j * radians * np.arange(len(burst)))
        burst = resample_poly(burst, 1, interp_rate)

    # Low-pass filter ------------------------------------------------------
    if not args.no_lpf:
        burst = lfilter(filter_taps, [1.0], burst)

    # Symbol-timing offset (STO) correction via cyclic-prefix correlation --
    interp_factor = 1
    burst = resample_poly(burst, interp_factor, 1)
    true_start = find_sto_cp(burst, sample_rate * interp_factor, legacy=legacy)
    burst = resample_poly(burst[true_start:], 1, interp_factor)

    # Coarse CFO correction via the first ZC's cyclic prefix ---------------
    zc_start_m = int(cp_schedule[:zc1_idx + 1].sum()) + fft_size * zc1_idx
    cfo_start = zc_start_m - short_cp_len - 1
    cfo_end = zc_start_m + fft_size - 1
    cfo_est_symbol = burst[cfo_start:cfo_end]
    cyclic_prefix = cfo_est_symbol[:short_cp_len]
    symbol_tail = cfo_est_symbol[-short_cp_len:]
    offset_radians = np.angle(np.vdot(cyclic_prefix, symbol_tail)) / fft_size
    burst = burst * np.exp(-1j * offset_radians * np.arange(1, len(burst) + 1))

    # Sub-sample fine timing & constant-phase correction (optional) --------
    data_carrier_indices_pre = get_data_carrier_indices(sample_rate)
    scale = sample_rate / 15.36e6
    if args.fine_timing:
        sub_sample_offset = find_zc_offset(
            burst, sample_rate, zc_idx=zc1_idx, zc_root=600,
            data_carrier_indices=data_carrier_indices_pre,
            search_range=15.0 * scale, n_steps=600, legacy=legacy)
        if sub_sample_offset != 0.0:
            print(f"  Sub-sample offset: {sub_sample_offset:+.4f}")
            burst = with_sample_offset(burst, sub_sample_offset)

    if args.fine_angle:
        constant_phase = find_zc_angle(burst, sample_rate,
                                        zc_idx=zc1_idx, legacy=legacy)
        if constant_phase != 0.0:
            burst = burst * np.exp(-1j * constant_phase)

    # OFDM symbol extraction and channel estimation ------------------------
    time_domain_syms, freq_domain_syms = extract_ofdm_symbol_samples(
        burst, sample_rate, legacy=legacy)

    channel1 = calculate_channel(freq_domain_syms[zc1_idx], sample_rate, 4)
    channel2 = calculate_channel(freq_domain_syms[zc2_idx], sample_rate, 6)
    channel1_data = channel1[data_carrier_indices]
    channel2_data = channel2[data_carrier_indices]
    channel1_phase = np.sum(np.angle(channel1_data)) / len(data_carrier_indices)
    channel2_phase = np.sum(np.angle(channel2_data)) / len(data_carrier_indices)
    _channel_phase_adj = (channel1_phase - channel2_phase) / 2  # noqa: F841
    channel = channel1_data

    if enable_plots:
        _plot_constellations(freq_domain_syms, data_carrier_indices, channel,
                              enable_equalizer, burst_idx + 1, _HERE,
                              num_symbols=num_symbols,
                              burst_label=burst_label)

    # QPSK demod + descramble + turbo decode (try 4 phase rotations) -------
    second_scrambler = generate_scrambler_seq(7200, scrambler_x2_init)
    bits_tmp = os.path.join(tempfile.gettempdir(), "droneid_bits")

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

        ones = int(np.sum(bits == 1))
        zeros = int(np.sum(bits == 0))
        if result.returncode == 0:
            print(f"  phase {phase_idx * 90}°: {ones} ones / {zeros} zeros → CRC OK")
            return result.stdout
        else:
            err = result.stderr.strip()
            print(f"  phase {phase_idx * 90}°: {ones} ones / {zeros} zeros → {err}")

    print(f"  Warning: all 4 phase rotations failed for {burst_label}")
    return ""


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
    sample_rate = args.sample_rate
    center_freq = args.center_freq

    # ------------------------------------------------------------------
    # Determine the list of target frequencies
    # ------------------------------------------------------------------
    if args.target_freq and args.target_freq != 0.0:
        target_freqs = [args.target_freq]
        print(f"Single-frequency mode: {args.target_freq / 1e6:g} MHz")
    else:
        target_freqs = load_in_band_marker_freqs(
            args.marker_freqs, center_freq, sample_rate)
        if not target_freqs:
            sys.exit(f"[ERROR] No marker frequencies in band "
                     f"[{(center_freq - sample_rate / 2) / 1e6:g}, "
                     f"{(center_freq + sample_rate / 2) / 1e6:g}] MHz. "
                     f"Check {args.marker_freqs} or pass --target-freq.")
        print(f"Multi-frequency mode: {len(target_freqs)} target(s) "
              f"in {sample_rate / 1e6:g} MHz band around "
              f"{center_freq / 1e6:g} MHz")

    # ------------------------------------------------------------------
    # Pre-computed constants
    # ------------------------------------------------------------------
    signal_bandwidth = 10e6
    filter_tap_count = 50
    filter_taps = firwin(filter_tap_count + 1, signal_bandwidth / sample_rate)

    fft_size = get_fft_size(sample_rate)
    structure = get_frame_structure(sample_rate, legacy=args.legacy)
    data_carrier_indices = get_data_carrier_indices(sample_rate)

    print(f"Frame layout: {'legacy (8 sym)' if args.legacy else 'modern (9 sym)'}, "
          f"ZC at 0-based indices "
          f"{structure['zc_symbol_indices'][0]},{structure['zc_symbol_indices'][1]}")

    _x2_pre_flip = np.array([
        0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0
    ], dtype=np.int32)
    scrambler_x2_init = _x2_pre_flip[::-1].copy()

    # ------------------------------------------------------------------
    # Multi-frequency ZC search (single file pass)
    # ------------------------------------------------------------------
    freq_to_indices = find_zc_indices_multi_freq(
        file_path, sample_rate, center_freq, target_freqs,
        args.threshold, chunk_duration_s=args.chunk_duration,
        sample_type=sample_type)

    total_bursts = sum(len(v) for v in freq_to_indices.values())
    if total_bursts == 0:
        sys.exit("[ERROR] No bursts found in the recording.")

    # ------------------------------------------------------------------
    # Per-frequency: extract bursts and decode each
    # ------------------------------------------------------------------
    frames_per_freq = {}
    for freq in target_freqs:
        indices = freq_to_indices.get(freq, np.array([], dtype=int))
        if len(indices) == 0:
            print(f"\n=== {freq / 1e6:.3f} MHz: no bursts ===")
            frames_per_freq[freq] = []
            continue

        print(f"\n=== {freq / 1e6:.3f} MHz: {len(indices)} burst(s) ===")
        freq_offset = center_freq - freq
        bursts = extract_bursts_at_indices(
            file_path, sample_rate, freq_offset, indices,
            padding=filter_tap_count, sample_type=sample_type,
            legacy=args.legacy)

        frames = []
        for burst_idx in range(bursts.shape[0]):
            burst = bursts[burst_idx].copy()
            burst_label = (f"{freq / 1e6:.3f} MHz "
                           f"burst {burst_idx + 1}/{bursts.shape[0]}")
            print(f"Burst {burst_label} @ sample offset {indices[burst_idx]}")
            frame = decode_burst(
                burst,
                sample_rate=sample_rate,
                args=args,
                filter_taps=filter_taps,
                filter_tap_count=filter_tap_count,
                structure=structure,
                fft_size=fft_size,
                data_carrier_indices=data_carrier_indices,
                scrambler_x2_init=scrambler_x2_init,
                turbo_decoder_path=turbo_decoder_path,
                burst_idx=burst_idx,
                burst_label=burst_label)
            frames.append(frame)
        frames_per_freq[freq] = frames

    # ------------------------------------------------------------------
    # Print decoded frames and parse to JSON
    # ------------------------------------------------------------------
    for freq in target_freqs:
        frames = frames_per_freq.get(freq, [])
        if not frames:
            continue
        for idx, frame in enumerate(frames):
            print(f"\nFRAME [{freq / 1e6:.3f} MHz #{idx + 1}]: {frame}", end="")
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

def _plot_time_spectrum(burst: np.ndarray, fs: float, idx: int,
                         label: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, num=43)
    axes[0].plot(10 * np.log10(np.abs(burst) ** 2 + 1e-30))
    title = "Time domain |x|² 10log10 (original)"
    if label:
        title += f"  [{label}]"
    axes[0].set_title(title)

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
                          num_symbols: int = 9,
                          burst_label: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
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
        title = f"Symbol {sym_idx + 1} IQ"
        if burst_label and sym_idx == 0:
            title += f"  [{burst_label}]"
        ax.set_title(title)
    plt.tight_layout()
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    fname = f"ofdm_symbol_{burst_num}.png"
    if burst_label:
        safe_label = burst_label.replace(" ", "_").replace("/", "of").replace(".", "p")
        fname = f"ofdm_{safe_label}.png"
    plt.savefig(os.path.join(img_dir, fname))
    plt.pause(0.01)


if __name__ == "__main__":
    main()
