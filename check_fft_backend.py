#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""FFT 백엔드 및 하드웨어 가속기 진단 스크립트.

Versal PetaLinux 환경에서 현재 사용 중인 FFT 구현체와
사용 가능한 하드웨어 가속기를 확인한다.

Usage:
    python3 check_fft_backend.py
    python3 check_fft_backend.py --bench      # 성능 벤치마크 포함
    python3 check_fft_backend.py --size 65536 # 벤치마크 FFT 크기 지정
"""

import sys
import os
import time
import importlib
import subprocess
import argparse
import platform
from pathlib import Path

import numpy as np

SEP = "-" * 60


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _try_import(name: str):
    try:
        return importlib.import_module(name), None
    except ImportError as e:
        return None, str(e)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return -1, f"명령을 찾을 수 없음: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def _bench(fn, label: str, n: int, repeats: int = 10) -> float:
    x = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    # warmup
    fn(x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    ms = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:<42} {ms:8.2f} ms/call")
    return ms


# ---------------------------------------------------------------------------
# 1. 시스템 환경
# ---------------------------------------------------------------------------

def check_system():
    print(SEP)
    print("[1] 시스템 환경")
    print(SEP)
    print(f"  Python       : {sys.version}")
    print(f"  플랫폼       : {platform.platform()}")
    print(f"  아키텍처     : {platform.machine()}")
    print(f"  CPU 코어 수  : {os.cpu_count()}")

    # /proc/cpuinfo 에서 CPU 모델 확인 (Linux)
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        lines = cpuinfo.read_text(errors="replace").splitlines()
        models = {l.split(":")[1].strip() for l in lines if l.startswith("model name")}
        for m in sorted(models):
            print(f"  CPU 모델     : {m}")

    # Versal 여부 확인
    for path in ["/proc/device-tree/compatible", "/sys/firmware/devicetree/base/compatible"]:
        p = Path(path)
        if p.exists():
            compat = p.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
            print(f"  DT compatible: {compat}")
            break


# ---------------------------------------------------------------------------
# 2. NumPy / SciPy FFT 백엔드
# ---------------------------------------------------------------------------

def check_numpy_scipy():
    print()
    print(SEP)
    print("[2] NumPy / SciPy FFT 백엔드")
    print(SEP)

    print(f"  numpy 버전   : {np.__version__}")

    # numpy FFT 내부 구현 확인
    np_fft_file = getattr(np.fft._pocketfft, "__file__", None) \
        if hasattr(np.fft, "_pocketfft") else None
    if np_fft_file:
        print(f"  numpy FFT    : pocketfft ({np_fft_file})")
    else:
        print(f"  numpy FFT    : {np.fft.__file__}")

    # scipy FFT 백엔드
    sft, err = _try_import("scipy.fft")
    if sft:
        import scipy
        print(f"  scipy 버전   : {scipy.__version__}")

        # get_backend()는 scipy 일부 버전에만 존재
        if hasattr(sft, "get_backend"):
            backend = sft.get_backend()
            print(f"  scipy 백엔드 : {backend}")
        else:
            # 내부 _uarray 백엔드 상태로 간접 확인
            try:
                import scipy.fft._pocketfft as _pf
                backend = f"pocketfft ({_pf.__file__})"
            except Exception:
                backend = "pocketfft (기본값 - get_backend 미지원 버전)"
            print(f"  scipy 백엔드 : {backend}")

        backend_file = getattr(sft, "__file__", None)
        if backend_file:
            print(f"  scipy.fft 경로: {backend_file}")
    else:
        print(f"  scipy.fft    : 없음 ({err})")


# ---------------------------------------------------------------------------
# 3. FFTW / pyfftw
# ---------------------------------------------------------------------------

def check_fftw() -> bool:
    print()
    print(SEP)
    print("[3] FFTW / pyfftw")
    print(SEP)

    pyfftw, err = _try_import("pyfftw")
    if pyfftw is None:
        print(f"  pyfftw       : 없음 ({err})")
        # 시스템 FFTW 라이브러리 직접 확인
        for lib in ["libfftw3.so", "libfftw3f.so", "libfftw3.so.3", "libfftw3f.so.3"]:
            rc, out = _run(["find", "/usr/lib", "/usr/local/lib", "/lib", "-name", lib])
            if rc == 0 and out:
                print(f"  시스템 FFTW  : {out.splitlines()[0]}")
                print("  → pyfftw 설치 시 scipy.fft 백엔드로 사용 가능: pip install pyfftw")
                return False
        print("  시스템 FFTW 라이브러리도 없음")
        return False

    print(f"  pyfftw 버전  : {pyfftw.__version__}")
    print(f"  FFTW 버전    : {pyfftw.fftw_version()}")
    print(f"  지원 타입    : {pyfftw.supported_types()}")

    # scipy 백엔드로 등록 가능한지 확인
    sft, _ = _try_import("scipy.fft")
    if sft:
        try:
            import pyfftw.interfaces.scipy_fft as pyfftw_sf
            with sft.set_backend(pyfftw_sf):
                # 교체 성공 여부를 실제 FFT 실행으로 확인
                x = np.ones(64, dtype=np.complex64)
                sft.fft(x)
            print("  scipy 백엔드 : pyfftw로 교체 가능 (set_backend 정상 동작)")
        except Exception as e:
            print(f"  scipy 백엔드 교체 : 실패 ({e})")
    return True


# ---------------------------------------------------------------------------
# 4. Xilinx / AMD Vitis XRT
# ---------------------------------------------------------------------------

def check_xrt():
    print()
    print(SEP)
    print("[4] Xilinx / AMD Vitis XRT (FPGA 가속기)")
    print(SEP)

    # xbutil 명령
    rc, out = _run(["xbutil", "examine"])
    if rc == 0:
        print("  xbutil examine: OK")
        for line in out.splitlines()[:15]:
            print(f"    {line}")
    else:
        print(f"  xbutil        : 없음 ({out[:80]})")

    # pyxrt Python 바인딩
    pyxrt, err = _try_import("pyxrt")
    if pyxrt:
        print(f"  pyxrt         : {getattr(pyxrt, '__version__', 'OK')}")
    else:
        print(f"  pyxrt         : 없음 ({err})")

    # XRT sysfs
    xrt_paths = [
        "/sys/bus/platform/drivers/zocl",
        "/sys/class/drm",
    ]
    for p in xrt_paths:
        exists = Path(p).exists()
        print(f"  {p}: {'있음' if exists else '없음'}")


# ---------------------------------------------------------------------------
# 5. AIE (AI Engine) - Versal 전용
# ---------------------------------------------------------------------------

def check_aie():
    print()
    print(SEP)
    print("[5] AIE (AI Engine) - Versal 전용")
    print(SEP)

    if sys.platform == "win32":
        print("  Windows 환경 - AIE/Linux sysfs 항목 건너뜀")
        return

    # AIE 디바이스 노드
    aie_devs = list(Path("/dev").glob("aie*")) if Path("/dev").exists() else []
    if aie_devs:
        print(f"  AIE 디바이스 : {[str(d) for d in aie_devs]}")
    else:
        print("  AIE 디바이스 : /dev/aie* 없음")

    # sysfs AIE 클래스
    aie_sysfs = Path("/sys/class/aie")
    if aie_sysfs.exists():
        entries = list(aie_sysfs.iterdir())
        print(f"  AIE sysfs    : {[e.name for e in entries]}")
        for e in entries:
            ver = e / "version"
            if ver.exists():
                print(f"    version: {ver.read_text().strip()}")
    else:
        print("  AIE sysfs    : /sys/class/aie 없음")

    # aiengine 커널 모듈
    rc, out = _run(["lsmod"])
    if rc == 0:
        aie_mods = [l for l in out.splitlines() if "aie" in l.lower() or "versal" in l.lower()]
        if aie_mods:
            print("  커널 모듈    :")
            for m in aie_mods:
                print(f"    {m}")
        else:
            print("  커널 모듈    : aie/versal 관련 모듈 없음")


# ---------------------------------------------------------------------------
# 6. OpenCL / OpenMP / BLAS
# ---------------------------------------------------------------------------

def check_misc():
    print()
    print(SEP)
    print("[6] 기타 가속 라이브러리")
    print(SEP)

    # PyOpenCL
    pyopencl, err = _try_import("pyopencl")
    if pyopencl:
        import pyopencl as cl
        platforms = cl.get_platforms()
        print(f"  PyOpenCL     : OK ({len(platforms)} 플랫폼)")
        for pf in platforms:
            for dev in pf.get_devices():
                print(f"    {pf.name} / {dev.name} ({cl.device_type.to_string(dev.type)})")
    else:
        print(f"  PyOpenCL     : 없음 ({err})")

    # numpy BLAS/LAPACK 정보
    try:
        info = np.show_config(mode="dicts")
        blas = info.get("Build Dependencies", {}).get("blas", {})
        if blas:
            print(f"  NumPy BLAS   : {blas.get('name','?')} {blas.get('version','')}")
    except Exception:
        pass

    # OpenMP 지원 확인 (scipy workers=-1 효과)
    try:
        import scipy
        fft_file = scipy.fft.__file__
        if sys.platform == "win32":
            # Windows: dumpbin으로 DLL 의존성 확인
            rc, out = _run(["dumpbin", "/dependents", fft_file])
            if rc == 0:
                omp_libs = [l.strip() for l in out.splitlines()
                            if "vcomp" in l.lower() or "omp" in l.lower()]
            else:
                # dumpbin 없으면 같은 폴더 DLL 목록으로 추정
                fft_dir = Path(fft_file).parent
                omp_libs = [str(p) for p in fft_dir.glob("*.pyd")
                            if "omp" in p.name.lower() or "gomp" in p.name.lower()]
        else:
            rc, out = _run(["ldd", fft_file])
            omp_libs = [l.strip() for l in (out or "").splitlines()
                        if "omp" in l.lower() or "gomp" in l.lower()]

        if omp_libs:
            print(f"  OpenMP (scipy): {omp_libs[0]}")
        else:
            print("  OpenMP (scipy): 미발견 → workers=-1 은 싱글스레드로 동작")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 7. 벤치마크
# ---------------------------------------------------------------------------

def run_benchmark(fft_size: int):
    print()
    print(SEP)
    print(f"[7] FFT 벤치마크  (size={fft_size:,}, complex64)")
    print(SEP)

    # numpy
    _bench(np.fft.fft, "numpy.fft.fft", fft_size)

    # scipy single / multi thread
    sft, _ = _try_import("scipy.fft")
    if sft:
        _bench(lambda x: sft.fft(x, workers=1),  "scipy.fft (workers=1)", fft_size)
        _bench(lambda x: sft.fft(x, workers=-1), f"scipy.fft (workers=-1, n={os.cpu_count()})", fft_size)

    # pyfftw
    pyfftw, _ = _try_import("pyfftw")
    if pyfftw:
        import pyfftw.interfaces.numpy_fft as pnp
        pyfftw.interfaces.cache.enable()
        _bench(pnp.fft, "pyfftw (numpy interface)", fft_size)

        threads = os.cpu_count()
        _bench(lambda x: pnp.fft(x, threads=threads),
               f"pyfftw (threads={threads})", fft_size)


# ---------------------------------------------------------------------------
# 8. 현재 process_live.py / correlator.py 가 쓰는 백엔드 요약
# ---------------------------------------------------------------------------

def print_summary():
    print()
    print(SEP)
    print("[요약] 현재 DroneID 파이프라인 FFT 경로")
    print(SEP)

    sft, _ = _try_import("scipy.fft")
    pyfftw_avail = _try_import("pyfftw")[0] is not None
    xrt_avail    = _try_import("pyxrt")[0] is not None

    if sft is None:
        backend = "scipy.fft 없음"
    elif pyfftw_avail:
        backend = "pyfftw (활성화 필요)"
    else:
        backend = "pocketfft (scipy 기본값)"

    print(f"  correlator.py  sft.fft(workers=-1)  → 백엔드: {backend}")
    print(f"  channel.py     np.fft.fft()          → 백엔드: pocketfft (numpy 내장)")
    print()
    if not pyfftw_avail and sft is not None:
        print("  [CPU] pocketfft 기반 소프트웨어 FFT만 사용 중 (FPGA 가속 없음)")
        print("  성능 개선 옵션:")
        print("    1. pip install pyfftw  -> FFTW3로 교체 (ARM에서 ~1.5-2x 빠름)")
        print("    2. Vitis HLS FFT IP + pyxrt 래퍼 작성 → PL(FPGA) 가속")
        print("    3. Versal AIE FFT 커널 → AIE 가속 (Vitis AI Engine 툴체인 필요)")
    elif backend != "scipy":
        print(f"  [{backend.upper()}] 가속 백엔드 활성화됨")
    if xrt_avail:
        print("  [XRT] pyxrt 설치됨 - Vitis FPGA 커널 호출 가능")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FFT 백엔드 및 하드웨어 가속기 진단")
    ap.add_argument("--bench", action="store_true", help="벤치마크 실행")
    ap.add_argument("--size",  type=int, default=65536,
                    help="벤치마크 FFT 크기 (기본값: 65536)")
    args = ap.parse_args()

    check_system()
    check_numpy_scipy()
    check_fftw()
    check_xrt()
    check_aie()
    check_misc()
    if args.bench:
        run_benchmark(args.size)
    print_summary()
    print()


if __name__ == "__main__":
    main()
