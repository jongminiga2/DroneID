"""20ms IQ → 1024×300 그레이 스펙트로그램 이미지 변환 (자급식, 전달용).

드론탐지 AI 모델(YOLO) 입력과 동일한 전처리. 가로 1024 = 주파수 빈,
세로 300 = 시간(20ms 워터폴). 122.88 MS/s complex IQ 기준.

이 파일은 Versal/디코더 측 전달용 자급식 버전 — repo 운영 경로(utils/iq.py:render_snap,
utils/dsp.py)와 수치적으로 동일한 로직을 의존성 없이 한 파일에 정리한 것.
운영 파이프라인을 바꾸면 여기도 함께 맞춰야 함.

⚠️ 핵심: viridis 컬러맵을 거쳐 '휘도'로 변환한다(직접 그레이 아님).
   직접 그레이는 약신호를 죽여 모델이 못 잡음 — 반드시 이 순서를 지킬 것.

의존: numpy, scipy, pillow, matplotlib
사용: python iq_to_image.py <iq_raw_int16le.bin> [out.png]
"""

import numpy as np
from scipy.fft import fft, fftshift
from PIL import Image
from matplotlib import colormaps

# ── 규격 (122.88 MS/s) ────────────────────────────────────────────
SR_HZ = 122.88e6
FFT = 8192  # = SR / 15kHz (OFDM 서브캐리어 간격)
AVG = 8  # 8192 → 1024 표시 빈
BINS = FFT // AVG  # 1024  (이미지 가로폭)
WF_MS = 20
ROWS = int(SR_HZ * WF_MS / 1000 / FFT)  # 300 (이미지 세로)
VIRIDIS = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def blackman_harris(n):
    k = np.arange(n, dtype=np.float32)
    t = (2.0 * np.pi / (n - 1)) * k
    return (
        0.35875
        - 0.48829 * np.cos(t)
        + 0.14128 * np.cos(2 * t)
        - 0.01168 * np.cos(3 * t)
    ).astype(np.float32)


def iq20ms_to_image(iq):
    """20ms complex IQ(>= 2,457,600 샘플) → 1024×300 그레이 PIL 이미지.

    iq: complex64 ndarray. 부족하면 None.
    """
    need = ROWS * FFT  # 2,457,600
    if len(iq) < need:
        return None
    x = iq[:need].reshape(ROWS, FFT)

    # 1) 윈도잉 + FFT + 파워 (coherent-gain 정규화)
    win = blackman_harris(FFT)
    cg = float(win.sum() / FFT)
    norm = 1.0 / (FFT * cg * cg)
    spec = fftshift(np.abs(fft(x * win, axis=1)) ** 2, axes=1)

    # 2) 8빈 평균 → 1024빈, PSD(dB)
    mag2 = spec.reshape(ROWS, BINS, AVG).mean(axis=2) * norm
    psd_db = 10.0 * np.log10(np.maximum(mag2, 1e-20))

    # 3) 고정 [-80, 0] dB → viridis 컬러맵
    nrm = np.clip((psd_db + 80.0) / 80.0, 0, 1)
    rgb = VIRIDIS[(nrm * 255).astype(np.uint8)]  # (300, 1024, 3)

    # 4) 퍼센타일(p2~p98) 대비 스트레치
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if hi - lo >= 1.0:
        rgb = np.clip((rgb.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(
            np.uint8
        )

    # 5) viridis → 휘도(그레이). PIL convert('L') = 학습 방식
    return Image.fromarray(rgb, "RGB").convert("L")  # 1024×300, mode 'L'


def i16le_interleaved_to_iq(raw_i16):
    """int16 LE interleaved [I0 Q0 I1 Q1 ...] → complex64 (±1.0)."""
    raw_i16 = raw_i16[: raw_i16.size & ~1]
    return (raw_i16.astype(np.float32).view(np.complex64) / 32768.0).astype(
        np.complex64
    )


if __name__ == "__main__":
    import sys

    # 예: PCIe로 받은 20ms IQ 원시 버퍼(int16 LE interleaved, 9,830,400 bytes)
    raw = np.fromfile(sys.argv[1], dtype="<i2")
    img = iq20ms_to_image(i16le_interleaved_to_iq(raw))
    img.save(sys.argv[2] if len(sys.argv) > 2 else "iq.png")  # 1024×300 PNG
