#!/usr/bin/env python3
"""
read_droneid.py

qdma_st.sh 기반으로 read_droneid.sh 기능을 Python으로 통합.
매 읽기 전 qdma501002 MM C2H 큐를 stop→del→add→start 재초기화하여
DMA stuck 없이 안정적으로 동작합니다.

사용법:
    sudo python3 read_droneid.py
    sudo python3 read_droneid.py --loop
    sudo python3 read_droneid.py --loop --interval 2.0
    sudo python3 read_droneid.py --no-setup          # 큐 초기 설정 생략
"""

import os
import sys
import signal
import struct
import argparse
import subprocess
import time
from datetime import datetime

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    import array as _array_mod
    _HAS_NUMPY = False

# ─────────────────────────────────────────────────────────────────────────────
# 상수 (qdma_st.sh / read_droneid.sh 동일)
# ─────────────────────────────────────────────────────────────────────────────

DEV_ST  = 'qdma501000'   # ST C2H ×4  (IQ 스트리밍용, 초기화만)
DEV_H2C = 'qdma501001'   # MM H2C ×1  (레지스터 제어용, 초기화만)
DEV_C2H = 'qdma501002'   # MM C2H ×1  (RSSI·탐지정보 DMA 읽기)

PCI_ST  = '0005:01:00.0'
PCI_H2C = '0005:01:00.1'
PCI_C2H = '0005:01:00.2'

ADDR_RSSI   = 0x800A00000
SIZE_RSSI   = 0x31          # 48 bytes ASCII + 최종 LF
ADDR_DETECT = 0x800A00040
SIZE_DETECT = 0x400         # 최대 26 레코드
ADDR_IQ     = 0x800009000
SIZE_IQ     = 9_830_400     # 4,915,200 int16 × 2 (I/Q 인터리브)

TMP_RSSI   = '/tmp/droneid_rssi.bin'
TMP_DETECT = '/tmp/droneid_detect.bin'
TMP_IQ     = '/tmp/droneid_iq.bin'
DEV_NODE   = f'/dev/{DEV_C2H}-MM-0'

PRODUCT_TYPES = {
     1:'Inspire 1',              2:'Phantom 3 Series',    3:'Phantom 3 Series',
     4:'Phantom 3 Std',          5:'M100',                6:'ACEONE',
     7:'WKM',                    8:'NAZA',                9:'A2',
    10:'A3',                    11:'Phantom 4',           12:'MG1',
    14:'M600',                  15:'Phantom 3 4k',        16:'Mavic Pro',
    17:'Inspire 2',             18:'Phantom 4 Pro',       20:'N2',
    21:'Spark',                 23:'M600 Pro',            24:'Mavic Air',
    25:'M200',                  26:'Phantom 4 Series',    27:'Phantom 4 Adv',
    28:'M210',                  30:'M210RTK',             31:'A3_AG',
    32:'MG2',                   34:'MG1A',                35:'Phantom 4 RTK',
    36:'Phantom 4 Pro V2.0',    38:'MG1P',                40:'MG1P-RTK',
    41:'Mavic 2',               44:'M200 V2 Series',      51:'Mavic 2 Enterprise',
    53:'Mavic Mini',            58:'Mavic Air 2',         59:'P4M',
    60:'M300 RTK',              61:'DJI FPV',             63:'Mini 2',
    64:'AGRAS T10',             65:'AGRAS T30',           66:'Air 2S',
    67:'M30',                   68:'Mavic 3',             69:'Mavic 2 Enterprise Advanced',
    70:'Mini SE',               72:'AGRAS T40',           73:'Mini 3 Pro',
    75:'DJI Avata',             76:'DJI Inspire 3',       77:'Mavic 3 Enterprise E/T/M',
    78:'DJI Flycart 30',        82:'AGRAS T25',           83:'AGRAS T50',
    84:'DJI Mavic 3 Pro',       86:'DJI Mavic 3 Classic', 87:'Mini 3',
    88:'DJI Mini 2 SE',         89:'M350 RTK',            90:'DJI Air 3',
    91:'DJI Matrice 3D/3TD',    93:'DJI Mini4 Pro',       95:'T60',
    96:'T25P',                 999:'unknown',
}

SEP  = '─' * 56
SEP2 = '=' * 56
RED  = '\033[91m'
RST  = '\033[0m'


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list, timeout: float = 8.0) -> bool:
    """dma-ctl 제어 명령. 별도 프로세스 그룹 + 타임아웃으로 무한 대기 방지."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            proc.wait(timeout=timeout)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return False
    except Exception:
        return False


def _run_shell(cmd: str) -> None:
    subprocess.run(cmd, shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _set_qmax(pci_bdf: str, n: int) -> None:
    """sysfs qmax 설정. 큐 ONLINE 상태에서 블록될 수 있으므로 5초 타임아웃."""
    path = f'/sys/bus/pci/devices/{pci_bdf}/qdma/qmax'
    try:
        proc = subprocess.Popen(
            ['sudo', 'tee', path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            proc.communicate(input=str(n).encode(), timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    except Exception:
        pass


def _dma_read(addr: int, size: int, outfile: str, timeout: float = 8.0) -> bool:
    """dma-from-device로 메모리 읽기. 타임아웃 초과 시 SIGKILL."""
    try:
        proc = subprocess.Popen(
            ['sudo', 'dma-from-device',
             '-d', DEV_NODE,
             '-a', hex(addr),
             '-s', hex(size),
             '-c', '1',
             '-f', outfile],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            proc.wait(timeout=timeout)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return False
    except Exception as e:
        print(f'  [DMA ERROR] {e}')
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. 초기 QDMA 설정  (qdma_st.sh 동일)
# ─────────────────────────────────────────────────────────────────────────────

def qdma_init() -> None:
    """qdma_st.sh 동일: 큐 정리 → qmax 설정 → 큐 추가 → 큐 시작.

    qmax 변경은 모든 큐가 DISABLED 상태일 때만 가능하므로
    stop/del 을 qmax 보다 먼저 수행한다.
    """
    print('[Init] 기존 큐 정리...')
    # ENABLED 큐는 stop이 즉시 실패(-22)하지만 del은 성공함.
    # ONLINE 큐는 커널 DMA 타임아웃(~10초)이 있으므로 stop에 15초 여유.
    # PCI FLR은 커널이 stop 실행 중일 때 호출하면 qdev=null → 패닉 유발하므로 사용 안 함.
    for idx in range(4):
        _run(['sudo', 'dma-ctl', DEV_ST,  'q', 'stop', 'idx', str(idx), 'dir', 'c2h'], timeout=15.0)
        _run(['sudo', 'dma-ctl', DEV_ST,  'q', 'del',  'idx', str(idx), 'dir', 'c2h'])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'q', 'stop', 'idx', '0', 'dir', 'h2c'], timeout=15.0)
    _run(['sudo', 'dma-ctl', DEV_H2C, 'q', 'del',  'idx', '0', 'dir', 'h2c'])
    _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'stop', 'idx', '0', 'dir', 'c2h'], timeout=15.0)
    if not _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'del', 'idx', '0', 'dir', 'c2h']):
        print('  [Init] 경고: C2H del 실패 (큐 ONLINE 잔존) — qmax 설정 건너뜀')

    print('[Init] qmax 설정...')
    _set_qmax(PCI_ST,  40)
    _set_qmax(PCI_H2C, 10)
    _set_qmax(PCI_C2H, 10)   # 큐 DISABLED 상태일 때만 적용됨 (ONLINE이면 무시됨)

    print('[Init] 큐 추가...')
    for idx in range(4):
        _run(['sudo', 'dma-ctl', DEV_ST,
              'q', 'add', 'idx', str(idx), 'mode', 'st', 'dir', 'c2h'])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'q', 'add', 'idx', '0', 'mode', 'mm', 'dir', 'h2c'])
    _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'add', 'idx', '0', 'mode', 'mm', 'dir', 'c2h'])

    print('[Init] 큐 시작...')
    _run(['sudo', 'dma-ctl', DEV_ST,  'q', 'start', 'list', '0', '3', 'dir', 'c2h'])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'q', 'start', 'idx',  '0', 'dir', 'h2c'])
    _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'start', 'idx',  '0', 'dir', 'c2h'])
    print('[Init] 완료.')


# ─────────────────────────────────────────────────────────────────────────────
# 2. 종료 시 모든 큐 정리
# ─────────────────────────────────────────────────────────────────────────────

def qdma_cleanup() -> None:
    """프로그램 종료 시 모든 QDMA 큐를 stop → del 하여 드라이버를 닫는다."""
    print('\n[Cleanup] QDMA 큐 정리 중...')
    _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'stop', 'idx', '0', 'dir', 'c2h'])
    _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'del',  'idx', '0', 'dir', 'c2h'])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'q', 'stop', 'idx', '0', 'dir', 'h2c'])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'q', 'del',  'idx', '0', 'dir', 'h2c'])
    for idx in range(4):
        _run(['sudo', 'dma-ctl', DEV_ST, 'q', 'stop', 'idx', str(idx), 'dir', 'c2h'])
        _run(['sudo', 'dma-ctl', DEV_ST, 'q', 'del',  'idx', str(idx), 'dir', 'c2h'])
    print('[Cleanup] 완료.')


# ─────────────────────────────────────────────────────────────────────────────
# 3. 매 읽기 전 MM C2H 큐 재초기화  (read_droneid.sh 상단 동일)
# ─────────────────────────────────────────────────────────────────────────────

def reinit_c2h() -> bool:
    """qdma501002 MM C2H 큐를 stop→del→add→start 로 재초기화.

    /dev/qdma501002-MM-0 은 q add(ENABLED) 시점에 생성되므로
    os.path.exists()로는 ONLINE 여부를 알 수 없다.
    q start 반환값으로 판단하며, PCI FLR은 사용하지 않는다.

    PCI FLR은 커널이 qdma_queue_stop 실행 중일 때 호출하면
    qdev가 NULL이 되어 다음 q add에서 커널 패닉을 유발한다.
    대신 q stop에 15초 타임아웃을 주어 커널이 자연 완료되도록 기다린다.
    """
    def _stop_del() -> None:
        # 커널 DMA 타임아웃이 ~10초이므로 15초 대기 후 확실히 완료
        _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'stop', 'idx', '0', 'dir', 'c2h'], timeout=15.0)
        _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'del',  'idx', '0', 'dir', 'c2h'])

    def _add_start() -> bool:
        _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'add',   'idx', '0', 'mode', 'mm', 'dir', 'c2h'])
        return _run(['sudo', 'dma-ctl', DEV_C2H, 'q', 'start', 'idx', '0', 'dir', 'c2h'])

    # 1차 시도
    _stop_del()
    if _add_start():
        return True

    # q start 실패 → 한 번 더 시도 (FLR 없이)
    print('  [reinit] q start 실패 → 재시도...')
    _stop_del()
    if _add_start():
        return True

    print('  [reinit] 경고: q start 실패 — DMA 불가')
    print('  [reinit] qdma 드라이버가 불안정한 경우 재부팅 필요: sudo reboot')
    return False


def _pci_flr(bdf: str) -> None:
    """PCI Function Level Reset — DMA 엔진 하드웨어 강제 리셋.

    sysfs reset 노드는 root 권한이 필요하므로 sudo tee 로 기록한다.
    """
    reset = f'/sys/bus/pci/devices/{bdf}/reset'
    if not os.path.exists(reset):
        print(f'  [FLR] {bdf}: reset 노드 없음 (FLR 미지원)')
        return
    ret = subprocess.run(
        f'echo 1 | sudo tee {reset}',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    if ret.returncode == 0:
        print(f'  [FLR] {bdf}: 완료')
    else:
        print(f'  [FLR] {bdf}: 실패 (sudo 권한 확인)')
    time.sleep(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RSSI 파싱  (read_droneid.sh 동일)
# ─────────────────────────────────────────────────────────────────────────────

def read_and_print_rssi(timeout: float = 8.0) -> bool:
    print(f'[DMA] RSSI 읽는 중... (최대 {timeout:.0f}초)', end=' ', flush=True)
    t0 = time.perf_counter()
    ok = _dma_read(ADDR_RSSI, SIZE_RSSI, TMP_RSSI, timeout=timeout)
    elapsed = time.perf_counter() - t0
    print(f'{"OK" if ok else "FAIL"} ({elapsed:.1f}s)')

    print()
    print(SEP)
    print(f'  RSSI / System Monitor  (0x{ADDR_RSSI:X})')
    print(SEP)
    if not ok or not os.path.exists(TMP_RSSI):
        print('  [ERROR] RSSI 읽기 실패')
        return False

    raw    = open(TMP_RSSI, 'rb').read()
    fields = raw.rstrip(b' \x00\x0a').split(b'\x0a')
    labels = ['RSSI            ', 'FPGA Temp (raw) ',
              'FPGA Volt (raw) ', 'AD9371 Temp(raw)']
    for label, val in zip(labels, fields):
        text = val.rstrip(b'\x0a\x0d').decode('ascii', errors='replace').strip()
        print(f'  {label}: {text}')

    try:
        os.remove(TMP_RSSI)
    except OSError:
        pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. IQ 읽기 및 통계  (mm_streaming.sh 동일)
# ─────────────────────────────────────────────────────────────────────────────

def _set_port(port: int) -> None:
    """RF 포트 선택 및 IQ 캡처 버퍼 리셋 (mm_streaming.sh 동일).

    port: 0=패턴, 1=포트1, 2=포트2
    """
    _run(['sudo', 'dma-ctl', DEV_H2C, 'reg', 'write', 'bar', '2', '0x4', str(port)])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'reg', 'write', 'bar', '2', '0x0', '0x0'])
    _run(['sudo', 'dma-ctl', DEV_H2C, 'reg', 'write', 'bar', '2', '0x0', '0x1'])


def _iq_stats(raw: bytes):
    """int16 인터리브 IQ 바이트 배열에서 통계를 계산.

    반환: (n_pairs, mean_i, mean_q, mean_abs_i, mean_abs_q, rms)
    """
    if _HAS_NUMPY:
        arr   = np.frombuffer(raw, dtype=np.int16)
        i_f   = arr[0::2].astype(np.float64)
        q_f   = arr[1::2].astype(np.float64)
        n     = len(i_f)
        m_i   = float(np.mean(i_f))
        m_q   = float(np.mean(q_f))
        ma_i  = float(np.mean(np.abs(i_f)))
        ma_q  = float(np.mean(np.abs(q_f)))
        rms   = float(np.sqrt(np.mean(i_f ** 2 + q_f ** 2)))
    else:
        buf  = _array_mod.array('h')
        buf.frombytes(raw)
        i_arr = buf[0::2]
        q_arr = buf[1::2]
        n     = len(i_arr)
        m_i   = sum(i_arr) / n
        m_q   = sum(q_arr) / n
        ma_i  = sum(abs(v) for v in i_arr) / n
        ma_q  = sum(abs(v) for v in q_arr) / n
        rms   = (sum(iv * iv + qv * qv for iv, qv in zip(i_arr, q_arr)) / n) ** 0.5
    return n, m_i, m_q, ma_i, ma_q, rms


def read_and_print_iq(port: int = 1, timeout: float = 15.0) -> bool:
    _set_port(port)
    print(f'[DMA] IQ 읽는 중... port={port}  (최대 {timeout:.0f}초)', end=' ', flush=True)
    t0 = time.perf_counter()
    ok = _dma_read(ADDR_IQ, SIZE_IQ, TMP_IQ, timeout=timeout)
    elapsed = time.perf_counter() - t0
    print(f'{"OK" if ok else "FAIL"} ({elapsed:.1f}s)')

    print()
    print(SEP)
    print(f'  IQ 통계  (0x{ADDR_IQ:X}, port={port})')
    print(SEP)
    if not ok or not os.path.exists(TMP_IQ):
        print('  [ERROR] IQ 읽기 실패')
        return False

    raw = open(TMP_IQ, 'rb').read()
    if len(raw) < 4:
        print('  데이터 없음')
        return False

    n, m_i, m_q, ma_i, ma_q, rms = _iq_stats(raw)
    print(f'  샘플 수 (I/Q 쌍) : {n:,}')
    print(f'  평균 I           : {m_i:+.2f}')
    print(f'  평균 Q           : {m_q:+.2f}')
    print(f'  평균 |I|         : {ma_i:.2f}')
    print(f'  평균 |Q|         : {ma_q:.2f}')
    print(f'  RMS              : {RED}{rms:.2f}{RST}')
    print(SEP)

    try:
        os.remove(TMP_IQ)
    except OSError:
        pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 6. 탐지정보 파싱  (read_droneid.sh 동일)
# ─────────────────────────────────────────────────────────────────────────────

def read_and_print_detect(timeout: float = 8.0) -> bool:
    print(f'[DMA] 탐지정보 읽는 중... (최대 {timeout:.0f}초)', end=' ', flush=True)
    t0 = time.perf_counter()
    ok = _dma_read(ADDR_DETECT, SIZE_DETECT, TMP_DETECT, timeout=timeout)
    elapsed = time.perf_counter() - t0
    print(f'{"OK" if ok else "FAIL"} ({elapsed:.1f}s)')
    print()
    print(SEP)
    print(f'  DroneID 탐지정보  (0x{ADDR_DETECT:X})')
    print(SEP)
    if not ok or not os.path.exists(TMP_DETECT):
        print('  [ERROR] 탐지정보 읽기 실패')
        return False

    data = open(TMP_DETECT, 'rb').read()
    if len(data) < 13:
        print('  데이터 없음')
        return False

    loop_count, ts, center_khz, tracking = struct.unpack_from('<IIIB', data, 0)
    ts_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else '-'
    print(f'  루프 번호     : #{loop_count}  ({ts_str})')
    print(f'  RF 중심주파수 : {RED}{center_khz:,} KHz  ({center_khz / 1000:.3f} MHz){RST}')
    print(f'  탐지 건수     : {tracking}')

    if tracking == 0:
        print('  (탐지된 드론 없음)')
    else:
        offset = 13
        for i in range(tracking):
            if offset + 39 > len(data):
                print(f'  #{i+1}: [데이터 부족]')
                break
            serial_b = data[offset:offset + 16]
            lat, lon, height, app_lat, app_lon, ptype, det_khz = \
                struct.unpack_from('<ffhffBI', data, offset + 16)
            offset += 39

            serial = serial_b.rstrip(b'\x00').decode('ascii', errors='replace')
            pname  = PRODUCT_TYPES.get(ptype, f'Unknown({ptype})')
            print(f'  ── #{i+1} ───────────────────────────────────────────')
            print(f'  Serial         : {serial}')
            print(f'  기종           : {pname}  (type={ptype})')
            print(f'  드론 위도      : {lat:.7f}')
            print(f'  드론 경도      : {lon:.7f}')
            print(f'  드론 고도      : {height} (raw int16)')
            print(f'  조종자 위도    : {app_lat:.7f}')
            print(f'  조종자 경도    : {app_lon:.7f}')
            print(f'  탐지 주파수    : {RED}{det_khz:,} KHz  ({det_khz / 1000:.3f} MHz){RST}')

    print(SEP)
    try:
        os.remove(TMP_DETECT)
    except OSError:
        pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 7. 인자 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='qdma_st.sh + read_droneid.sh 통합 Python 스크립트\n'
                    '매 읽기 전 qdma501002 MM C2H 큐를 자동 재초기화합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--loop',      action='store_true',
                   help='Ctrl-C 까지 반복 실행')
    p.add_argument('--interval',  type=float, default=1.0, metavar='SECS',
                   help='반복 간격 (초, 기본 1.0)')
    p.add_argument('--no-setup',  action='store_true',
                   help='초기 qdma_init() 생략 (이미 큐가 설정된 경우)')
    p.add_argument('--dma-timeout', type=float, default=8.0, metavar='SECS',
                   help='RSSI/탐지 DMA 최대 대기 시간 (초, 기본 8.0)')
    p.add_argument('--port', type=int, default=1, metavar='N',
                   help='IQ RF 포트 (0=패턴 1=포트1 2=포트2, 기본 1)')
    p.add_argument('--no-iq', action='store_true',
                   help='IQ 읽기 생략')
    p.add_argument('--iq-timeout', type=float, default=15.0, metavar='SECS',
                   help='IQ DMA 최대 대기 시간 (초, 기본 15.0)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 8. 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if os.geteuid() != 0:
        sys.exit('[ERROR] root 권한 필요 — sudo python3 read_droneid.py 로 실행하세요.')

    # ── 초기 QDMA 설정 (1회) ────────────────────────────────────────────────
    if not args.no_setup:
        qdma_init()

    iteration = 0
    mode_str  = f'루프 (간격 {args.interval:.1f}초)' if args.loop else '단일'
    print(f'\n{SEP2}')
    print(f'  DroneID 모니터  —  {mode_str}')
    print(SEP2)

    try:
        while True:
            iteration += 1
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'\n{SEP2}')
            print(f'  [{now}]  반복 #{iteration}')
            print(SEP2)

            # ── 매 읽기 전 MM C2H 큐 재초기화 ──────────────────────────────
            print('[Queue] MM C2H 재초기화...')
            if not reinit_c2h():
                print('  [경고] 큐 초기화 실패 — 이번 회차 DMA 건너뜀')
                if not args.loop:
                    break
                print(f'\n  다음 실행까지 {args.interval:.1f}초 대기...')
                time.sleep(args.interval)
                continue

            # ── RSSI 읽기 + 출력 ─────────────────────────────────────────
            read_and_print_rssi(args.dma_timeout)

            # ── 탐지정보 읽기 + 출력 ─────────────────────────────────────
            read_and_print_detect(args.dma_timeout)

            # ── IQ 읽기 + 통계 출력 ──────────────────────────────────────
            if not args.no_iq:
                print('[Queue] IQ용 MM C2H 재초기화...')
                if reinit_c2h():
                    read_and_print_iq(args.port, args.iq_timeout)
                else:
                    print('  [경고] 큐 초기화 실패 — IQ 건너뜀')

            if not args.loop:
                break

            print(f'\n  다음 실행까지 {args.interval:.1f}초 대기...')
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print('\n[INFO] 사용자 중단 (Ctrl-C).')
    finally:
        for f in (TMP_RSSI, TMP_DETECT, TMP_IQ):
            try:
                os.remove(f)
            except OSError:
                pass
        if not args.no_setup:
            qdma_cleanup()


if __name__ == '__main__':
    main()
