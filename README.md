# DroneID

DJI DroneID 버스트를 IQ 녹음 파일에서 검출/복조/디코드하는 Python 파이프라인입니다. MATLAB `process_file.m`의 포팅 버전입니다.

## 처리 흐름

1. 타깃 주파수 목록 결정 (단일 지정 또는 `marker_freqs.json` 자동 스캔)
2. ZC 상관기로 버스트 탐색
3. 버스트 추출
4. 정수/Coarse CFO 보정
5. OFDM 심볼 복조
6. 디스크램블 후 `cpp/remove_turbo`로 Turbo 디코드
7. 프레임을 hex로 출력하고 `parse_frame.py`로 JSON 파싱

## 요구 사항

- Python 3 (numpy, scipy, matplotlib)
- `cpp/remove_turbo` 바이너리가 사전 컴파일되어 있어야 함 (Windows에서는 `remove_turbo.exe`)

## 사용법

```
python process_file.py --file <IQ파일> [옵션...]
```

### 두 가지 동작 모드

**1) 단일 주파수 모드** — `--target-freq`를 명시
```
python process_file.py --file recording.bin \
    --sample-type int16 --center-freq 2.4595e9 --target-freq 2.4595e9
```

**2) 다중 주파수 스캔 모드** (기본값) — `marker_freqs.json`에서 `center_freq ± sample_rate/2` 범위에 들어오는 모든 주파수를 자동으로 스캔
```
python process_file.py --file recording.bin \
    --sample-type int16 --center-freq 2.45e9
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--file` | (필수) | IQ 녹음 파일 경로 |
| `--sample-type` | `int16` | 샘플 타입: `single`, `float32`, `int16`, `int8` 등 |
| `--sample-rate` | `122.88e6` | 녹음 샘플 레이트 (Hz) |
| `--center-freq` | `0` | SDR 중심 주파수 (Hz) |
| `--target-freq` | `0` | DroneID 신호 주파수. 0이면 다중 스캔 모드 |
| `--threshold` | `0.7` | ZC 상관 임계값 (0.0~1.0) |
| `--chunk-duration` | `0.680` | 상관기 청크 크기 (초) |
| `--marker-freqs` | `marker_freqs.json` | 마커 주파수 JSON 경로 |

### 디코드 동작 플래그

| 플래그 | 설명 |
|---|---|
| `--legacy` | 8-symbol 레거시 프레임 디코드 (Mavic Pro / Mavic 2) |
| `--fine-timing` | 서브샘플 타이밍 보정 (`find_zc_offset`) |
| `--fine-angle` | ZC DC-bin 위상 보정 (`find_zc_angle`) |
| `--ifo` | 업샘플링 IFO 주파수 오프셋 보정 (target_freq가 7.5kHz 이상 어긋날 때만 사용) |
| `--no-equalizer` | 주파수 도메인 이퀄라이저 비활성화 |
| `--no-lpf` | 저역 통과 필터 단계 생략 |
| `--no-plots` | matplotlib 플롯 비활성화 |

## 샘플 녹음 사용 예

```
# Mavic 2 (legacy 8-symbol)
python process_file.py --file ../samples/c2440p0_t2444p5_mavic2.bin --center-freq 2440.0e6 --target-freq 2444.5e6 --no-plots --legacy

python process_file.py --file ../samples/c2450p0_t2414p5_mavic2.bin --center-freq 2450.0e6 --target-freq 2414.5e6 --no-plots --legacy

# Mavic Pro / Mini 3 / Air 2S / Mini 2 GPS (modern 9-symbol)
python process_file.py --file ../samples/c2440p0_t2429p5_mavicpro.bin --center-freq 2440.0e6 --target-freq 2429.5e6 --no-plots

python process_file.py --file ../samples/c2440p0_t2429p5_mini3.bin --center-freq 2440.0e6 --target-freq 2429.5e6 --no-plots

python process_file.py --file ../samples/c2440p0_t2459p5_air2s.bin --center-freq 2440.0e6 --target-freq 2459.5e6 --no-plots

python process_file.py --file ../samples/c2450p0_t2444p5_mini2gps.bin --center-freq 2450.0e6 --target-freq 2444.5e6 --no-plots

# 다중 주파수 스캔 (target-freq 생략 → marker_freqs.json 사용)
python process_file.py --file ../samples/c2440p0_t2429p5_mini3.bin --center-freq 2440.0e6 --no-plots
```

## 관련 파일

- `marker_freqs.json` — 스캔 대상 주파수 목록 (MHz)
- `parse_frame.py` — 디코드된 hex 프레임을 JSON으로 파싱
- `cpp/remove_turbo` — Turbo 디코더 (C++ 빌드 산출물)
