"""시험장 부정 기기 신호 이상 탐지 PoC 공통 정의 (2026-11-18).

스냅샷 한 장 = 20 MS/s IQ 20 ms(400,000 표본) → 스펙트로그램 → 주파수 256줄 × 시간 97칸 → 배경 잡음 세기 기준 dB → 고정 눈금 uint8 이미지.
  - 수신 조건: HackRF One, 중심 2,449.5 MHz, 폭 20 MHz(2,439.5–2,459.5 MHz), 20 MS/s.
  - 배경 노이즈 = 공원에서 드론·조종기를 끄고 받은 IQ 데이터 3초(rf-drone-detector 저장소 릴리스 hackrf_1_train_background.tar 의
    3_neo2_park/02_telescopic_OFF_near1m 세션, CC BY 4.0)에서 블루투스 모양 덩어리를 지운 것.
    data/background/Park02_{1,2,3}.npy (1초씩, complex64, 이 저장소 릴리스에서 내려받음): 1 = 학습셋, 2 = 검증셋, 3 = 테스트셋.
    세 파일은 시간이 겹치지 않는다.
  - SNR = 신호의 1 Hz 당 세기가 배경 잡음 세기의 1 Hz 당 세기보다 몇 dB 큰가 (rf-drone-detector 저장소의 IQ 합성 코드와 같은 정의).
    SNR 0 dB 신호가 있는 칸은 잡음만 있는 칸보다 약 3 dB 밝다.
  - 이미지 밝기는 스냅샷마다 맞추지 않고 고정 눈금(-6 dB = 0, +34 dB = 255)을 쓴다. 픽셀차를 구하려면 스냅샷끼리 같은 눈금이어야 하기 때문이다
    (rf-drone-detector 의 그림 함수 to_image 는 스냅샷마다 1–99.5 백분위로 밝기를 맞추므로 여기서는 쓰지 않는다).
경로는 이 파일 위치 기준 상대 경로로만 만든다.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]                       # 저장소 맨 위
BG_DIR = ROOT / "data" / "background"
BG_FILE = {"train": "Park02_1.npy", "val": "Park02_2.npy", "test": "Park02_3.npy"}
DATA, RUNS, FIG = ROOT / "data", ROOT / "runs", ROOT / "doc" / "fig"
MODELS, RESULTS = ROOT / "models", ROOT / "results"              # 공개한 모델 파일, 기준 결과(비교용)

FS, WIN = 20e6, 0.02
N = int(FS * WIN)                                                # 400,000 표본
FC = 2449.5e6                                                    # 수신 중심 주파수
NFFT, HOP = 8192, 4081                                           # rf-drone-detector 그림과 같은 격자: 주파수 칸 2.44 kHz, 시간 칸 204 µs
POOL = 32                                                        # 주파수 32칸 평균 → 한 줄 78.1 kHz
NF, NT = NFFT // POOL, (N - NFFT) // HOP + 1                     # 256줄, 97칸
DF_ROW, DT_COL = FS / NF, HOP / FS                               # 한 줄 78.125 kHz, 한 칸 204 µs
DB_LO, DB_HI = -6.0, 34.0                                        # 이미지 눈금
DC_HALF = 41                                                     # 0 Hz ±100 kHz(41칸)는 바로 바깥 칸 값으로 채운다(rf-drone-detector spectrogram_v7 과 같은 처리)

_W = np.hanning(NFFT).astype(np.float32)
_WSUM = float(np.sum(_W.astype(np.float64) ** 2))
_k0 = NFFT // 2
_SRC = np.arange(NFFT)
for _d in range(-DC_HALF + 1, DC_HALF):
    _SRC[_k0 + _d] = _k0 + (_d + DC_HALF if _d >= 0 else _d - DC_HALF)


def spec_db(x):
    """20 ms IQ → dB 스펙트로그램 [256 주파수줄(아래 = 낮은 주파수), 97 시간칸]. 잡음 전력 1 이면 잡음만 있는 칸의 기댓값 0 dB."""
    x = np.asarray(x, dtype=np.complex64)
    x = x - x.mean()
    fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP][:NT] * _W
    P = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=1), axes=1)) ** 2 / _WSUM
    P = P[:, _SRC]
    return (10 * np.log10(P.reshape(NT, NF, POOL).mean(2) + 1e-12)).T.astype(np.float32)


def to_u8(db):
    return np.clip((db - DB_LO) / (DB_HI - DB_LO) * 255 + 0.5, 0, 255).astype(np.uint8)


def to_db(u8):
    return np.asarray(u8, dtype=np.float32) / 255 * (DB_HI - DB_LO) + DB_LO


# ---------------- 배경 노이즈 ----------------
_BG = {}


def background(role):
    """역할(train/val/test)의 공원 무신호 IQ(mmap)와 그 rms. 잡음 전력을 1 로 맞출 때 쓴다."""
    if role not in _BG:
        a = np.load(BG_DIR / BG_FILE[role], mmap_mode="r")
        v = np.asarray(a)
        _BG[role] = (a, float(np.sqrt(np.mean(np.abs(v - v.mean()) ** 2))))
    return _BG[role]


def bg_snapshot(role, rng):
    a, sd = background(role)
    off = int(rng.integers(0, len(a) - N + 1))
    return (np.asarray(a[off: off + N]) / sd).astype(np.complex64), off


# ---------------- 신호 덩어리 ----------------
def band_burst(n, f_lo, f_hi, rng, rms):
    """[f_lo, f_hi](창 가운데 기준 Hz) 안에만 성분이 있는 잡음 덩어리. rf-drone-detector 의 band_burst 와 같은 식."""
    F = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    fr = np.fft.fftfreq(n, 1 / FS)
    F[(fr < f_lo) | (fr > f_hi)] = 0
    b = np.fft.ifft(F)
    return (b / np.sqrt(np.mean(np.abs(b) ** 2) + 1e-20) * rms).astype(np.complex64)


def rms_for(snr_db, bw):
    """1 Hz 당 세기가 배경 잡음 세기(전력 1, 폭 FS)보다 snr_db 크도록 하는 덩어리 rms."""
    return float(np.sqrt(10 ** (snr_db / 10) / FS * bw))


def add_burst(x, rng, f_lo, f_hi, t0, t1, snr_db):
    """창 안에 보이는 부분만 더한다. 반환 박스 (t0, t1, f_lo, f_hi) 또는 None(안 보임)."""
    f_lo, f_hi = max(f_lo, -FS / 2), min(f_hi, FS / 2)
    s0, s1 = max(0, int(round(t0 * FS))), min(N, int(round(t1 * FS)))
    if s1 - s0 < 200 or f_hi - f_lo < 100e3:
        return None
    x[s0:s1] += band_burst(s1 - s0, f_lo, f_hi, rng, rms_for(snr_db, f_hi - f_lo))
    return (s0 / FS, s1 / FS, f_lo, f_hi)


# ---------------- 정상 신호 (아주 작게) ----------------
WIFI_BW = 16.6e6
WIFI_VISIBLE_CH = [c for c in range(1, 14) if abs(2412e6 + 5e6 * (c - 1) - FC) < FS / 2 + WIFI_BW / 2 - 1e6]   # 창에 보이는 와이파이 채널(5–12)
NORMAL = dict(n_wifi=(0, 2), snr_wifi=(2.0, 8.0), dur_wifi=(0.1e-3, 3e-3),
              n_bt=(0, 4), snr_bt=(3.0, 10.0))


def add_normal(x, rng):
    """정상 신호: 와이파이 0–2개(폭 16.6 MHz, 0.1–3 ms, SNR 2–8 dB) + 블루투스 짧은 패킷 0–4개(SNR 3–10 dB).
    블루투스는 절반은 클래식(폭 1 MHz, 0.366 ms = 한 슬롯 패킷), 절반은 BLE(폭 2 MHz, 0.08–0.376 ms = 광고·짧은 데이터 패킷).
    채널은 창 안에 보이는 것만 고른다. 반환 박스 목록 [(종류, t0, t1, f_lo, f_hi)]."""
    boxes = []
    for _ in range(int(rng.integers(NORMAL["n_wifi"][0], NORMAL["n_wifi"][1] + 1))):
        c = int(rng.choice(WIFI_VISIBLE_CH)); fc = 2412e6 + 5e6 * (c - 1) - FC
        dur = rng.uniform(*NORMAL["dur_wifi"]); t0 = rng.uniform(-dur / 2, WIN - dur / 2)
        b = add_burst(x, rng, fc - WIFI_BW / 2, fc + WIFI_BW / 2, t0, t0 + dur, rng.uniform(*NORMAL["snr_wifi"]))
        if b: boxes.append(("wifi",) + b)
    for _ in range(int(rng.integers(NORMAL["n_bt"][0], NORMAL["n_bt"][1] + 1))):
        if rng.random() < 0.5:
            ch = int(rng.integers(38, 58)); fc = 2402e6 + ch * 1e6 - FC; bw, dur = 1e6, 0.366e-3          # 2,440–2,459 MHz
        else:
            k = int(rng.integers(19, 29)); fc = 2402e6 + 2e6 * k - FC; bw, dur = 2e6, rng.uniform(0.08e-3, 0.376e-3)   # 2,440–2,458 MHz
        t0 = rng.uniform(-dur / 2, WIN - dur / 2)
        b = add_burst(x, rng, fc - bw / 2, fc + bw / 2, t0, t0 + dur, rng.uniform(*NORMAL["snr_bt"]))
        if b: boxes.append(("bt",) + b)
    return boxes


# ---------------- 학습·검증용 이상: 모양을 정하지 않은 사각형 에너지 덩어리 ----------------
BLOB = dict(n=(1, 4), bw=(0.3e6, 4e6), dur=(0.3e-3, 6e-3), snr=(0.0, 12.0))


def add_blobs(x, rng):
    """1–4개, 폭 0.3–4 MHz, 길이 0.3–6 ms, 위치 무작위(창 안), SNR 0–12 dB. 주파수 도약 같은 구조는 없다."""
    boxes = []
    for _ in range(int(rng.integers(BLOB["n"][0], BLOB["n"][1] + 1))):
        bw = rng.uniform(*BLOB["bw"]); dur = rng.uniform(*BLOB["dur"])
        fc = rng.uniform(-FS / 2 + bw / 2, FS / 2 - bw / 2); t0 = rng.uniform(0, WIN - dur)
        b = add_burst(x, rng, fc - bw / 2, fc + bw / 2, t0, t0 + dur, rng.uniform(*BLOB["snr"]))
        if b: boxes.append(("blob",) + b)
    return boxes


# ---------------- 테스트용 이상: 처음 보는 블루투스 2종 ----------------
# 1) 블루투스 이어폰으로 소리를 보내는 신호(휴대폰 → 이어폰, 클래식 블루투스 오디오 전송):
#    긴 패킷(3-DH5, 2.87 ms)이 3.75 ms(6 슬롯)마다, 패킷마다 79개 채널(2,402–2,480 MHz, 1 MHz 간격) 중 하나로 옮겨 다닌다.
# 2) 저전력 블루투스(BLE) 연속 데이터 전송(AI 안경·BLE 오디오 흉내): 연결 간격 7.5 ms 마다 한 채널(데이터 채널 37개, 2 MHz 간격)에서
#    긴 패킷(LE 2M 251바이트, 1.06 ms) 2개를 0.33 ms 사이를 두고 보낸다.
# 수신 폭이 20 MHz 라 블루투스 채널의 약 1/4 만 보인다 → 기기가 켜져 있어도 스냅샷에 아무것도 안 보일 수 있다(n_vis = 0).
ANOM = {1: dict(name="블루투스 이어폰 오디오 전송", bw=1e6, pkt=2.87e-3, period=3.75e-3),
        2: dict(name="BLE 연속 데이터 전송", bw=2e6, pkt=1.06e-3, gap=0.33e-3, period=7.5e-3)}
BLE_DATA_CH = [k for k in range(40) if k not in (0, 12, 39)]      # 2,402 + 2k MHz 중 광고 채널(2,402·2,426·2,480 MHz) 3개를 뺀 37개


def add_anomaly(x, rng, kind, snr_db):
    """반환 (박스 목록, 보낸 패킷 수, 보이는 패킷 수)."""
    a = ANOM[kind]; boxes = []; n_tx = 0
    ph = rng.uniform(0, a["period"])
    for k in range(-1, int(WIN / a["period"]) + 2):
        t0 = ph + k * a["period"]
        if kind == 1:
            pk = [(t0, t0 + a["pkt"])]; ch = int(rng.integers(0, 79)); fc = 2402e6 + ch * 1e6 - FC
        else:
            pk = [(t0, t0 + a["pkt"]), (t0 + a["pkt"] + a["gap"], t0 + 2 * a["pkt"] + a["gap"])]
            fc = 2402e6 + 2e6 * int(rng.choice(BLE_DATA_CH)) - FC
        for (p0, p1) in pk:
            if p1 <= 0 or p0 >= WIN: continue
            n_tx += 1
            if abs(fc) > FS / 2: continue
            b = add_burst(x, rng, fc - a["bw"] / 2, fc + a["bw"] / 2, p0, p1, snr_db)
            if b: boxes.append((f"anom{kind}",) + b)
    return boxes, n_tx, len(boxes)


# ---------------- 장소별 실험 (2026-11-19) ----------------
# 환경 변수 EXAM_SITE 가 비어 있으면 11-18 실험 그대로(공원 배경, data/·runs/). indoor, indoor_droneoff, park 중 하나면
# site_setup.py 의 그 장소 설정(중심 주파수, 배경 파일, 잡음 세기 맞추기, data/site/<장소>/·runs/site/<장소>/)으로 바꾼다.
import os as _os
SITE = _os.environ.get("EXAM_SITE", "")
SITE_NAME = ""
if SITE:
    import site_setup as _site_setup
    _site_setup.apply(globals())
