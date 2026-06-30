#!/usr/bin/env python3
"""DJI DroneID live IQ processor – captures IQ from an AD9371 (IIO/libiio)
and runs the same burst-decode pipeline as process_file.py.

Usage:
    # Hardware (marker_freqs.json 자동 스캔)
    python process_live.py --center-freq 2.44e9

    # 복수 중심 주파수 (순서대로 각 1회 처리)
    python process_live.py --center-freq 2.44e9 2.45e9 2.46e9

    # 복수 중심 주파수 루프 (모든 CF를 순환하며 반복)
    python process_live.py --center-freq 2.44e9 2.45e9 2.46e9 --loop

    # 특정 IP / 특정 주파수
    python process_live.py --ip 192.168.0.24 --center-freq 2.44e9 \\
        --target-freq 2.4295e9

    # 시뮬레이션 모드 (libiio 없어도 동작)
    python process_live.py --sim --center-freq 2.44e9

    # 연속 루프
    python process_live.py --center-freq 2.44e9 --loop
"""

import sys
import os
import json
import struct
import argparse
import subprocess
import time
from typing import List

import numpy as np
from scipy.signal import firwin

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from process_file import decode_burst, load_in_band_marker_freqs
from utils import get_fft_size, get_data_carrier_indices, get_frame_structure
from correlator import find_zc_indices_from_array
from burst_extractor import extract_bursts_from_array

try:
    from parse_frame import parse_frame as _parse_frame_fn
except ImportError:
    _parse_frame_fn = None

# ---------------------------------------------------------------------------
# IIO / AD9371 device layer  (GUI 의존성 없이 직접 구현)
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 122.88e6
PHY_DEV_NAME   = 'adrv9371-phy'
RX_DEV_NAME    = 'axi-adrv9371-rx-hpc'

try:
    import iio  # type: ignore
    _IIO_AVAILABLE = True
except (ImportError, OSError, TypeError) as _e:
    _IIO_AVAILABLE = False
    print(f'[INFO] libiio not available ({_e.__class__.__name__}) → simulation mode')


class AD9371Device:
    def __init__(self):
        self.ctx     = None
        self.phy     = None
        self.rx      = None
        self.rx_buf  = None
        self._i_chs  = []
        self._q_chs  = []
        self._buf_samples = None

    @staticmethod
    def list_devices(uri: str) -> List[str]:
        if not _IIO_AVAILABLE:
            raise RuntimeError('pylibiio 미설치 — pip install pylibiio')
        return [d.name for d in iio.Context(uri).devices if d.name]

    def connect(self, uri: str,
                phy_name: str = PHY_DEV_NAME,
                rx_name:  str = RX_DEV_NAME) -> None:
        if not _IIO_AVAILABLE:
            raise RuntimeError('pylibiio 미설치 — pip install pylibiio')
        self.ctx = iio.Context(uri)
        self.phy = self.ctx.find_device(phy_name)
        self.rx  = self.ctx.find_device(rx_name)
        if self.phy is None or self.rx is None:
            names = [d.name for d in self.ctx.devices if d.name]
            raise RuntimeError(
                f"PHY '{phy_name}' 또는 RX '{rx_name}' 찾을 수 없음.\n"
                f"Available: {names}")

    def disconnect(self) -> None:
        self.rx_buf = None
        self.ctx = self.phy = self.rx = None
        self._i_chs = []
        self._q_chs = []

    def _phy_attr(self, ch_id: str, candidates: List[str],
                  value: str, warn: bool = True) -> None:
        try:
            ch = next(c for c in self.phy.channels if c.id == ch_id)
        except StopIteration:
            return
        for attr in candidates:
            if attr in ch.attrs:
                try:
                    ch.attrs[attr].value = value
                    return
                except Exception as exc:
                    if warn:
                        print(f'[WARN] {ch_id}.{attr}={value}: {exc}')

    def set_lo(self, center_hz: float) -> None:
        self._phy_attr('altvoltage0',
                       ['frequency', 'RX_LO_frequency'],
                       str(int(center_hz)))

    def configure_rf(self, center_hz: float,
                     bw_hz: float, agc_mode: str) -> None:
        self.set_lo(center_hz)
        for cid in ('voltage0', 'voltage1'):
            self._phy_attr(cid, ['rf_bandwidth', 'bandwidth'],
                           str(int(bw_hz)), warn=False)
            self._phy_attr(cid, ['gain_control_mode', 'agc_mode'],
                           agc_mode, warn=False)

    @staticmethod
    def _split_iq(scan_chs):
        try:
            I_MOD = iio.ChannelModifier.IIO_MOD_I
            Q_MOD = iio.ChannelModifier.IIO_MOD_Q
            i_chs = [c for c in scan_chs if c.modifier == I_MOD]
            q_chs = [c for c in scan_chs if c.modifier == Q_MOD]
            if i_chs and q_chs:
                return i_chs, q_chs
        except Exception:
            pass
        i_chs = [c for c in scan_chs if (c.id or '').lower().endswith(('_i', 'i'))]
        q_chs = [c for c in scan_chs if (c.id or '').lower().endswith(('_q', 'q'))]
        if i_chs and q_chs and len(i_chs) == len(q_chs):
            return i_chs, q_chs
        return scan_chs[0::2], scan_chs[1::2]

    def setup_buffer(self, buf_samples: int) -> None:
        if self._buf_samples == buf_samples and self.rx_buf is not None:
            return
        all_scan = sorted(
            [c for c in self.rx.channels if not c.output and c.scan_element],
            key=lambda c: c.index)
        if not all_scan:
            raise RuntimeError(f"'{self.rx.name}' scan_element 채널 없음")
        i_chs, q_chs = self._split_iq(all_scan)
        hw = min(2, min(len(i_chs), len(q_chs)))
        if hw < 1:
            raise RuntimeError('I/Q 채널 쌍 부족')
        use_i = i_chs[:hw]
        use_q = q_chs[:hw]
        use_set = {id(c) for c in use_i + use_q}
        self.rx_buf = None
        for c in all_scan:
            c.enabled = (id(c) in use_set)
        self._i_chs = use_i
        self._q_chs = use_q
        self._buf_samples = buf_samples
        self.rx_buf = iio.Buffer(self.rx, buf_samples, False)

    def capture(self):
        """1회 refill → (raw_i, raw_q) int16 ndarray (RX1)"""
        self.rx_buf.refill()
        raw_i = np.frombuffer(
            self._i_chs[0].read(self.rx_buf), dtype='<i2').copy()
        raw_q = np.frombuffer(
            self._q_chs[0].read(self.rx_buf), dtype='<i2').copy()
        return raw_i, raw_q


class SimDevice:
    """libiio 미설치 / 디바이스 없을 때 사용하는 더미 IQ 생성기."""
    def __init__(self):
        self._buf_samples = 65536
        self._center_hz   = 2440e6
        self._sample_rate = SAMPLE_RATE_HZ

    def connect(self, *_a, **_kw):        pass
    def disconnect(self):                  pass
    def set_lo(self, center_hz: float):   self._center_hz = center_hz

    def configure_rf(self, center_hz: float, bw_hz: float,
                     agc_mode: str) -> None:
        self._center_hz = center_hz

    def setup_buffer(self, buf_samples: int) -> None:
        self._buf_samples = buf_samples

    def capture(self):
        n = self._buf_samples
        t = np.arange(n) / self._sample_rate
        sig = (0.6  * np.exp(1j * 2 * np.pi * 5e6  * t)
               + 0.25 * np.exp(1j * 2 * np.pi * -20e6 * t)
               + 0.05 * (np.random.randn(n) + 1j * np.random.randn(n)))
        sig *= 16384.0
        raw_i = np.clip(sig.real, -32768, 32767).astype('<i2')
        raw_q = np.clip(sig.imag, -32768, 32767).astype('<i2')
        time.sleep(n / self._sample_rate)
        return raw_i, raw_q


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
_MARKER_FREQS_JSON = os.path.join(_HERE, 'marker_freqs.json')
# IIO 단일 refill 크기.
# ip:127.0.0.1 경로: 소켓 대역폭 ~50MB/s → 4M 샘플(32MB) × 6회 ≈ 3.8s
# local: 경로:  DMA 대역폭 ~20MB/s → 버퍼 크기 무관하게 ~9.8s (더 느림)
# 따라서 이 보드에서는 ip:127.0.0.1 + 큰 버퍼가 최적.
_IIO_CHUNK = 4_194_304

# ---------------------------------------------------------------------------
# Memory map (devmem) – write_icd.xlsx 기반, ip:127.0.0.1 시에만 사용
# ---------------------------------------------------------------------------
# RSSI / 시스템 모니터 블록 (write_rssi.sh 동일 주소)
_RSSI_BASE   = 0x800A00000   # RSSI + FPGA온도 + FPGA전압 + AD9371온도 (48 bytes)
_RSSI_END    = 0x800A00030   # 0x0A (LF) 최종 기록 위치 (스크립트 동일)

# 탐지정보 블록 (RSSI 블록 이후 번지, 8-byte 정렬)
_DETECT_BASE = 0x800A00040

# IIO sysfs 경로 (write_rssi.sh 동일)
_SYSFS_RSSI        = '/sys/bus/iio/devices/iio:device2/in_voltage0_rssi'
_SYSFS_FPGA_TEMP   = '/sys/bus/iio/devices/iio:device0/in_temp160_temp_raw'
_SYSFS_FPGA_VOLT   = '/sys/bus/iio/devices/iio:device0/in_voltage5_vcc_soc_raw'
_SYSFS_AD9371_TEMP = '/sys/bus/iio/devices/iio:device2/in_temp_raw'

# 탐지정보 레코드 크기: serial_no[16]+lat(4)+lon(4)+height(2)+app_lat(4)+app_lon(4)+product_type(1)+det_freq(4) = 39
_DETECT_RECORD_SIZE = 39

# parse_frame 구조체에서 device_type(uint8)의 바이트 오프셋
# <BBBHH16siihhhhhhQiiiiBB...  →  1+1+1+2+2+16+4+4+2+2+2+2+2+2+8+4+4+4+4 = 67
_DEVICE_TYPE_OFFSET = 67


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='DJI DroneID live IQ processor (AD9371 IIO, 콘솔 전용)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group('Device')
    g.add_argument('--ip', default='127.0.0.1',
                   help='AD9371 IP 주소 (iiod). 빈 문자열 입력 시 local: 직접 접근')
    g.add_argument('--phy', default=PHY_DEV_NAME,
                   help='IIO PHY 디바이스 이름')
    g.add_argument('--rx',  default=RX_DEV_NAME,
                   help='IIO RX 디바이스 이름')
    g.add_argument('--sim', action='store_true',
                   help='시뮬레이션 모드 강제 (하드웨어 불필요)')
    g.add_argument('--rf-bw', type=float, default=100e6,
                   help='RF 대역폭 (Hz)')
    g.add_argument('--agc', default='slow_attack',
                   choices=['slow_attack', 'fast_attack', 'hybrid', 'manual'],
                   help='AGC 모드')

    g = p.add_argument_group('Capture')
    g.add_argument('--capture-ms', type=float, default=200.0,
                   help='1회 캡처 시간 (ms)')
    g.add_argument('--loop', action='store_true',
                   help='Ctrl-C 까지 반복 캡처·처리')

    g = p.add_argument_group('RF')
    g.add_argument('--sample-rate', type=float, default=SAMPLE_RATE_HZ,
                   help='샘플레이트 (Hz)')
    g.add_argument('--center-freq', type=float, nargs='+', required=True,
                   help='SDR 중심 주파수 (Hz), 여러 개 지정 가능. 예: 2.44e9 2.45e9')
    g.add_argument('--target-freq', type=float, default=0.0,
                   help='DroneID 신호 주파수 (Hz); 0 이면 marker_freqs.json 사용')

    g = p.add_argument_group('Processing (process_file.py 동일)')
    g.add_argument('--threshold',      type=float, default=0.7)
    g.add_argument('--chunk-duration', type=float, default=0.680,
                   help='상관기 청크 크기 (초)')
    g.add_argument('--marker-freqs',   default=_MARKER_FREQS_JSON)
    g.add_argument('--no-equalizer',   action='store_true')
    g.add_argument('--no-plots',       action='store_true',
                   help='(항상 True — 라이브 모드에서 플롯 비활성화)')
    g.add_argument('--legacy',         action='store_true',
                   help='레거시 8-심볼 프레임 (Mavic Pro / Mavic 2)')
    g.add_argument('--fine-timing',    action='store_true')
    g.add_argument('--fine-angle',     action='store_true')
    g.add_argument('--ifo',            action='store_true')
    g.add_argument('--no-lpf',         action='store_true')

    args = p.parse_args()
    args.no_plots = True  # 라이브 모드는 항상 플롯 없음
    return args


# ---------------------------------------------------------------------------
# Device connection
# ---------------------------------------------------------------------------

def _connect(args: argparse.Namespace, center_freq: float):
    use_sim = args.sim or not _IIO_AVAILABLE
    if use_sim:
        dev = SimDevice()
        dev.connect()
        dev.configure_rf(center_freq, args.rf_bw, args.agc)
        print('[Device] 시뮬레이션 모드')
        return dev

    # 같은 보드라면 'local:' 이 iiod 네트워크 스택을 우회해 훨씬 빠릅니다.
    # --ip 인수가 비어 있으면 local: 을 사용합니다.
    uri = 'local:' if not args.ip else f'ip:{args.ip}'
    dev = AD9371Device()
    try:
        dev.connect(uri, phy_name=args.phy, rx_name=args.rx)
    except RuntimeError:
        names = AD9371Device.list_devices(uri)
        nl = [(n, n.lower()) for n in names]
        phy = next((n for n in names if 'phy' in n.lower()), None)
        rx  = next(
            (n for n, s in nl
             if 'rx' in s and 'tx' not in s and 'obs' not in s), None)
        if not phy or not rx:
            raise RuntimeError(f'PHY/RX 자동 감지 실패. Available: {names}')
        dev.connect(uri, phy_name=phy, rx_name=rx)
        args.phy = phy
        args.rx  = rx
        print(f'[Device] 자동 감지: PHY={phy}  RX={rx}')

    dev.configure_rf(center_freq, args.rf_bw, args.agc)
    print(f'[Device] 연결됨  URI={uri}  '
          f'CF={center_freq/1e6:.3f} MHz  '
          f'BW={args.rf_bw/1e6:.0f} MHz  AGC={args.agc}')
    return dev


def _setup_buffer(dev, args: argparse.Namespace) -> None:
    total = int(args.capture_ms * 1e-3 * args.sample_rate)
    chunk = min(total, _IIO_CHUNK)
    chunk = max(4096, chunk // 4096 * 4096)
    dev.setup_buffer(chunk)
    print(f'[Device] IIO 버퍼: {chunk:,} 샘플 '
          f'(~{chunk/args.sample_rate*1000:.1f} ms/refill, '
          f'{-(-total // chunk)} 회 refill)')


# ---------------------------------------------------------------------------
# devmem / memory write helpers (ip:127.0.0.1 전용)
# ---------------------------------------------------------------------------

def _devmem_write_bytes(base_addr: int, data: bytes) -> None:
    """data를 base_addr 부터 1바이트씩 devmem으로 기록 (write_rssi.sh 방식)."""
    for i, b in enumerate(data):
        subprocess.run(
            ['devmem', hex(base_addr + i), '8', hex(b)],
            check=False, capture_output=True)


def _write_rssi_to_mem() -> None:
    """write_rssi.sh 기능을 Python으로 구현.
    RSSI, FPGA온도, FPGA전압, AD9371온도를 sysfs에서 읽어 _RSSI_BASE 에 기록.
    48 bytes 를 채우고 _RSSI_END(0x800A00030)에 0x0A 한 바이트를 추가 기록."""
    paths = [_SYSFS_RSSI, _SYSFS_FPGA_TEMP, _SYSFS_FPGA_VOLT, _SYSFS_AD9371_TEMP]
    buf = bytearray()
    for path in paths:
        try:
            val = open(path, 'rb').read().rstrip(b'\n\r')
        except OSError:
            val = b'N/A'
        buf.extend(val)
        buf.append(0x0A)          # LF 구분자

    region = _RSSI_END - _RSSI_BASE   # 0x30 = 48 bytes
    if len(buf) < region:
        buf.extend(b'\x20' * (region - len(buf)))   # 공백 패딩
    else:
        buf = buf[:region]

    _devmem_write_bytes(_RSSI_BASE, bytes(buf))
    # 스크립트와 동일하게 _RSSI_END 에 최종 0x0A 기록
    subprocess.run(['devmem', hex(_RSSI_END), '8', '0x0a'],
                   check=False, capture_output=True)


def _write_center_freq_to_mem(center_freq_hz: float) -> None:
    """설정 중심 주파수를 탐지정보 헤더의 rf_centerfrequency 필드에 즉시 기록.
    Header layout: loop_count(4B) + timestamp(4B) + rf_centerfrequency(4B)"""
    center_khz = max(0, int(center_freq_hz / 1000))
    data = struct.pack('<I', center_khz)
    _devmem_write_bytes(_DETECT_BASE + 8, data)
    print(f'[Memory] CF 설정: {center_khz} KHz → 0x{_DETECT_BASE + 8:X}')


def _write_detect_to_mem(center_freq_hz: float, detections: list,
                         loop_count: int = 0) -> None:
    """write_icd.xlsx '탐지정보' 레이아웃으로 _DETECT_BASE 에 기록.

    Header (13 bytes):
        loop_count         : uint32_t  (루프 반복 번호)
        timestamp          : uint32_t  (Unix 시각, 초)
        rf_centerfrequency : uint32_t  (KHz)
        tracking_number    : uint8_t   (레코드 수)
    Per record (39 bytes, #1~연속):
        serial_no[16] + lat(f) + lon(f) + height(h) +
        app_lat(f) + app_lon(f) + product_type(B) + det_freq(I, KHz)
    """
    center_khz = max(0, int(center_freq_hz / 1000))
    n = min(len(detections), 255)
    ts = int(time.time())
    buf = struct.pack('<IIIB', loop_count, ts, center_khz, n)

    for det in detections[:n]:
        serial = det['serial_number'].encode('ascii', errors='replace')[:16]
        serial = serial + b'\x00' * (16 - len(serial))
        raw_h = max(-32768, min(32767,
                    int(round(det.get('height_m', 0.0) * 3.281))))
        det_khz = max(0, int(det.get('detection_freq_hz', 0) / 1000))
        buf += serial
        buf += struct.pack('<ffhffBI',
                           det.get('latitude', 0.0),
                           det.get('longitude', 0.0),
                           raw_h,
                           det.get('app_lat', 0.0),
                           det.get('app_lon', 0.0),
                           det.get('product_type_raw', 0) & 0xFF,
                           det_khz)

    _devmem_write_bytes(_DETECT_BASE, buf)
    ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    print(f'[Memory] Loop #{loop_count}  {ts_str}')
    print(f'[Memory] 탐지정보 {n}건 → 0x{_DETECT_BASE:X}  '
          f'(CF={center_khz} KHz, {len(buf)} bytes)')


# ---------------------------------------------------------------------------
# IQ capture
# ---------------------------------------------------------------------------

def _capture_iq(dev, total_samples: int) -> np.ndarray:
    """total_samples 개의 IQ 수신 → int16 인터리브 flat 배열 (I0 Q0 I1 Q1 …)"""
    i_acc = np.empty(total_samples, dtype='<i2')
    q_acc = np.empty(total_samples, dtype='<i2')
    done = 0
    while done < total_samples:
        raw_i, raw_q = dev.capture()
        take = min(len(raw_i), len(raw_q), total_samples - done)
        i_acc[done:done + take] = raw_i[:take]
        q_acc[done:done + take] = raw_q[:take]
        done += take
    flat = np.empty(total_samples * 2, dtype='<i2')
    flat[0::2] = i_acc
    flat[1::2] = q_acc
    return flat


# ---------------------------------------------------------------------------
# Single-window processing
# ---------------------------------------------------------------------------

def _process_once(dev, args: argparse.Namespace, *,
                  center_freq: float,
                  turbo_decoder_path: str,
                  target_freqs: List[float],
                  filter_taps: np.ndarray,
                  filter_tap_count: int,
                  fft_size: int,
                  structure: dict,
                  data_carrier_indices: np.ndarray,
                  scrambler_x2_init: np.ndarray,
                  use_devmem: bool = False,
                  detections: dict = None,
                  loop_count: int = 0) -> None:

    sample_rate   = args.sample_rate
    total_samples = int(args.capture_ms * 1e-3 * sample_rate)
    t_start = time.perf_counter()

    # ── 0. RSSI / 시스템 모니터 메모리 기록 (ip:127.0.0.1 전용) ─────────────
    if use_devmem:
        _write_rssi_to_mem()
        if detections is not None:
            detections.clear()   # 매 캡처 주기마다 이전 탐지 초기화

    # ── 1. IQ 캡처 ──────────────────────────────────────────────────────────
    if use_devmem:
        # _write_center_freq_to_mem(0.0)   # 캡처 시작 전: LO = 0 기록
        time.sleep(1.0)                   # 1초 대기 후 캡처
    t0 = time.perf_counter()
    flat_iq = _capture_iq(dev, total_samples)
    print(f'[Capture]   {total_samples:,} 샘플  '
          f'{(time.perf_counter()-t0)*1000:.0f} ms')
    if use_devmem:
        _write_center_freq_to_mem(center_freq)   # 캡처 완료 후: 실제 LO 값 복원

    # ── 2. ZC 상관 (인메모리 FFT 기반, 임시 파일 없음) ──────────────────────
    t0 = time.perf_counter()
    freq_to_indices = find_zc_indices_from_array(
        flat_iq, sample_rate, center_freq, target_freqs,
        args.threshold)
    print(f'[Correlate] {(time.perf_counter()-t0)*1000:.0f} ms  '
          f'({sum(len(v) for v in freq_to_indices.values())} 버스트 발견)')

    if not any(len(v) for v in freq_to_indices.values()):
        if use_devmem:
            _write_detect_to_mem(center_freq, [], loop_count)
        print(f'[Total]     {(time.perf_counter()-t_start)*1000:.0f} ms  → 버스트 없음')
        return

    # ── 3. 버스트 추출 + 디코드 (인메모리) ──────────────────────────────────
    frames_per_freq: dict = {}
    for freq in target_freqs:
        indices = freq_to_indices.get(freq, np.array([], dtype=int))
        if len(indices) == 0:
            print(f'\n=== {freq/1e6:.3f} MHz: 버스트 없음 ===')
            frames_per_freq[freq] = []
            continue

        print(f'\n=== {freq/1e6:.3f} MHz: {len(indices)} 버스트 ===')
        bursts = extract_bursts_from_array(
            flat_iq, sample_rate, center_freq - freq, indices,
            padding=filter_tap_count, legacy=args.legacy)

        frames = []
        for i in range(bursts.shape[0]):
            burst = bursts[i].copy()
            label = f'{freq/1e6:.3f} MHz burst {i+1}/{bursts.shape[0]}'
            print(f'Burst {label} @ sample {indices[i]}')
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
                burst_idx=i,
                burst_label=label)
            frames.append(frame)
        frames_per_freq[freq] = frames

    # ── 4. 디코드 결과 출력 + 탐지정보 메모리 기록 ──────────────────────────
    for freq in target_freqs:
        for idx, frame in enumerate(frames_per_freq.get(freq, [])):
            print(f'\nFRAME [{freq/1e6:.3f} MHz #{idx+1}]: {frame}', end='')
            frame_hex = frame.strip()
            if not frame_hex:
                continue

            parsed = None
            if _parse_frame_fn is not None:
                try:
                    parsed = _parse_frame_fn(frame_hex)
                    print(json.dumps(parsed, indent=4, ensure_ascii=False))
                except Exception as exc:
                    print(f'Warning: parse_frame 실패: {exc}')
            else:
                ret = subprocess.run(
                    [sys.executable,
                     os.path.join(_HERE, 'parse_frame.py'),
                     frame_hex],
                    capture_output=True, text=True)
                if ret.returncode == 0:
                    print(ret.stdout)
                    try:
                        parsed = json.loads(ret.stdout)
                    except Exception:
                        parsed = None
                else:
                    print(f'Warning: parse_frame.py 실패 (exit {ret.returncode})')

            # 탐지정보 누적 (ip:127.0.0.1 전용)
            if use_devmem and detections is not None and parsed is not None:
                raw_bytes = bytes.fromhex(frame_hex)
                product_type_raw = (raw_bytes[_DEVICE_TYPE_OFFSET]
                                    if len(raw_bytes) > _DEVICE_TYPE_OFFSET else 0)
                sn = parsed.get('serial_number') or f'unk_{idx}'
                detections[sn] = {
                    'serial_number':     sn,
                    'latitude':          parsed.get('latitude', 0.0),
                    'longitude':         parsed.get('longitude', 0.0),
                    'height_m':          parsed.get('height_m', 0.0),
                    'app_lat':           parsed.get('app_lat', 0.0),
                    'app_lon':           parsed.get('app_lon', 0.0),
                    'product_type_raw':  product_type_raw,
                    'detection_freq_hz': freq,
                }

    # 탐지정보 전체 목록을 메모리에 기록 (#1~연속)
    if use_devmem:
        _write_detect_to_mem(center_freq, list(detections.values()), loop_count)

    print(f'[Total]     {(time.perf_counter()-t_start)*1000:.0f} ms')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    center_freqs: List[float] = args.center_freq  # nargs='+' → 리스트

    turbo_decoder_path = os.path.join(_HERE, 'cpp', 'remove_turbo')
    if sys.platform.startswith('win') and not os.path.isfile(turbo_decoder_path):
        turbo_decoder_path += '.exe'
    if not os.path.isfile(turbo_decoder_path):
        sys.exit(f"[ERROR] remove_turbo를 찾을 수 없음: '{turbo_decoder_path}'\n"
                 "        C++ 소스를 먼저 컴파일하세요.")

    # 중심 주파수별 타겟 주파수 결정
    cf_targets: List[tuple] = []   # [(center_freq, [target_freqs...]), ...]
    for cf in center_freqs:
        if args.target_freq and args.target_freq != 0.0:
            tf = [args.target_freq]
            print(f'CF={cf/1e6:g} MHz  → 단일 타겟: {args.target_freq/1e6:g} MHz')
        else:
            tf = load_in_band_marker_freqs(args.marker_freqs, cf, args.sample_rate)
            if not tf:
                print(f'[WARN] CF={cf/1e6:g} MHz: 대역 내 마커 주파수 없음, 건너뜀 '
                      f'[{(cf - args.sample_rate/2)/1e6:g}, '
                      f'{(cf + args.sample_rate/2)/1e6:g}] MHz')
                continue
            print(f'CF={cf/1e6:g} MHz  → {len(tf)} 타겟 주파수')
        cf_targets.append((cf, tf))

    if not cf_targets:
        sys.exit(f'[ERROR] 유효한 중심 주파수가 없습니다. '
                 f'{args.marker_freqs} 확인 또는 --target-freq 지정.')

    # 디코드 상수 사전 계산
    filter_tap_count = 50
    filter_taps      = firwin(filter_tap_count + 1, 10e6 / args.sample_rate)
    fft_size         = get_fft_size(args.sample_rate)
    structure        = get_frame_structure(args.sample_rate, legacy=args.legacy)
    data_carrier_indices = get_data_carrier_indices(args.sample_rate)

    print(f"프레임 구조: {'레거시 8-심볼' if args.legacy else '모던 9-심볼'}  "
          f"ZC @ 인덱스 "
          f"{structure['zc_symbol_indices'][0]},{structure['zc_symbol_indices'][1]}")

    _x2_pre_flip = np.array([
        0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0
    ], dtype=np.int32)
    scrambler_x2_init = _x2_pre_flip[::-1].copy()

    # 디바이스 연결 (첫 번째 CF로 초기화)
    dev = _connect(args, cf_targets[0][0])
    _setup_buffer(dev, args)

    # ip:127.0.0.1 일 때만 devmem 쓰기 활성화 (시뮬레이션 모드 제외)
    use_devmem = (args.ip == '127.0.0.1') and not args.sim
    detections: dict = {}   # serial_number → 최신 탐지 레코드 (#1~연속 유지)
    if use_devmem:
        print(f'[Memory] devmem 쓰기 활성화'
              f'  RSSI=0x{_RSSI_BASE:X}  탐지정보=0x{_DETECT_BASE:X}')

    cf_count = len(cf_targets)
    mode_str = ('루프' if args.loop else '단일') + (f' ({cf_count}개 CF 순환)' if cf_count > 1 else '')
    print(f'[Mode] {mode_str}')

    kwargs = dict(
        turbo_decoder_path=turbo_decoder_path,
        filter_taps=filter_taps,
        filter_tap_count=filter_tap_count,
        fft_size=fft_size,
        structure=structure,
        data_carrier_indices=data_carrier_indices,
        scrambler_x2_init=scrambler_x2_init,
        use_devmem=use_devmem,
        detections=detections,
    )

    try:
        iteration = 0
        while True:
            iteration += 1
            for cf, tf in cf_targets:
                dev.set_lo(cf)
                print(f'\n{"="*60}')
                print(f'[{time.strftime("%H:%M:%S")}] 반복 {iteration}  '
                      f'CF={cf/1e6:.3f} MHz  '
                      f'캡처={args.capture_ms:.0f} ms')
                print('=' * 60)
                _process_once(dev, args, center_freq=cf, target_freqs=tf,
                              **kwargs, loop_count=iteration)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print('\n[INFO] 사용자 중단.')
    finally:
        try:
            dev.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    main()
