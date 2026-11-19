"""장소별 실험 설정 (2026-11-19). 환경 변수 EXAM_SITE 로 장소를 고르면 common.py 가 맨 끝에서 이 설정으로 바꾼다.

장소마다 그 장소에서 받은 정상 IQ 데이터만으로 모델을 학습하고, 같은 장소의 뒤쪽 데이터로 시험한다(장소마다 따로 학습).
| EXAM_SITE        | 장소          | IQ 데이터 (1초 파일)                                        | 중심 주파수 | 학습셋 | 검증셋 | 테스트셋 |
| indoor           | 실내          | D4D 1_nodrone_indoor/20261007_antenna_on, 5개 (.cs8)       | 2,450.0 MHz | 1–3번 | 4번   | 5번     |
| indoor_droneoff  | 실내(드론 끔) | D4D 2_neo2_indoor/20261008_drone_off, 5개 (.cs8)           | 2,450.5 MHz | 1–3번 | 4번   | 5번     |
| park             | 공원          | data/background/Park02_1·2·3.npy (11-18 실험과 같은 파일)   | 2,449.5 MHz | 1번   | 2번   | 3번     |
파일 번호는 받은 순서다(실내는 약 4초 간격으로 1초씩 받음). 앞 파일로 학습하고 뒤 파일로 시험해서, 장소에서 미리 받은 데이터로 학습한 뒤
시험 중에 추론하는 흐름을 따른다. 실내 두 세션은 같은 방이지만 날짜·중심 주파수·수신 필터(15 MHz, 20 MHz)·수집 도구가 다르다.

잡음 세기 맞추기: 학습셋 파일마다 |x|² 의 중앙값 ÷ ln 2 (복소 가우스 잡음이면 평균 전력과 같다)를 구해 평균한 값 하나를 그 장소의 잡음 전력으로
정하고, 검증셋·테스트셋에도 같은 값을 쓴다. 11-18 실험은 데이터셋마다 평균 전력으로 맞췄다. 그런데 실내(드론 끔) 2번 파일에는 20 MHz 전체를 덮는
아주 센 신호(잡음보다 35 dB 넘게 셈, 약 120 ms)가 있어 평균 전력이 잡음의 27배가 되므로 중앙값을 쓴다. 이 신호는 그 장소의 정상 신호로 보고 지우지 않는다.

배경 IQ 파일 만들기(실내 두 곳, .cs8 → complex64 .npy):
  1) rf-drone-detector 저장소 v1.0 릴리스의 hackrf_3_test_indoor.tar(CC BY 4.0)를 저장소 맨 위에서 푼다 → data/raw/hackrf/1_nodrone_indoor/…, 2_neo2_indoor/…
  2) python src/site_setup.py [--hackrf data/raw/hackrf]   (경로는 저장소 맨 위 기준 상대 경로. 연구 폴더에서는 --hackrf ../../contest/d4d_2026/data/raw/hackrf)
  → data/background/<장소>/01.npy … 05.npy
경로는 이 파일 위치 기준 상대 경로로만 만든다.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = os.environ.get("EXAM_SITE", "")
BG = ROOT / "data" / "background"
RAW = {"indoor": "1_nodrone_indoor/20261007_antenna_on", "indoor_droneoff": "2_neo2_indoor/20261008_drone_off"}
SITES = {
    "indoor": dict(name="실내", fc=2450.0e6, session="D4D 1_nodrone_indoor/20261007_antenna_on (10-07, 안테나 끼움, 드론 없음)",
                   files={"train": ["indoor/01.npy", "indoor/02.npy", "indoor/03.npy"], "val": ["indoor/04.npy"], "test": ["indoor/05.npy"]}),
    "indoor_droneoff": dict(name="실내(드론 끔)", fc=2450.5e6, session="D4D 2_neo2_indoor/20261008_drone_off (10-08, 드론·조종기 끔)",
                            files={"train": [f"indoor_droneoff/0{k}.npy" for k in (1, 2, 3)], "val": ["indoor_droneoff/04.npy"], "test": ["indoor_droneoff/05.npy"]}),
    "park": dict(name="공원", fc=2449.5e6, session="D4D 3_neo2_park/02_telescopic_OFF_near1m (10-08, 드론·조종기 끔, 블루투스 모양 덩어리를 지운 11-18 파일)",
                 files={"train": ["Park02_1.npy"], "val": ["Park02_2.npy"], "test": ["Park02_3.npy"]}),
}
ORDER = ["indoor", "indoor_droneoff", "park"]


def noise_power(files):
    """파일마다 |x - 평균|² 의 중앙값 ÷ ln 2 를 구해 평균한다. 센 신호가 잠깐 있어도 배경 잡음 세기 값이 거의 바뀌지 않는다."""
    v = []
    for f in files:
        x = np.asarray(np.load(f, mmap_mode="r")); x = x - x.mean()
        v.append(float(np.median(x.real ** 2 + x.imag ** 2) / np.log(2)))
    return float(np.mean(v)), v


def apply(g):
    """common.py 의 전역 이름을 이 장소 설정으로 바꾼다(g = common 의 globals())."""
    s = SITES[SITE]
    g["SITE_NAME"] = s["name"]
    g["FC"] = s["fc"]
    g["WIFI_VISIBLE_CH"] = [c for c in range(1, 14) if abs(2412e6 + 5e6 * (c - 1) - g["FC"]) < g["FS"] / 2 + g["WIFI_BW"] / 2 - 1e6]
    g["DATA"] = ROOT / "data" / "site" / SITE
    g["RUNS"] = ROOT / "runs" / "site" / SITE
    g["FIG"] = ROOT / "doc" / "fig" / "site"
    files = {r: [BG / f for f in fs] for r, fs in s["files"].items()}
    state = {}

    def background(role):
        """역할(train/val/test)의 배경 IQ 파일 목록(mmap)과 장소 잡음 전력의 제곱근(학습셋 파일로 정한 값 하나)."""
        if "sd" not in state:
            p, per = noise_power(files["train"]); state["sd"] = float(np.sqrt(p)); state["per_file"] = per
        if role not in state:
            state[role] = [np.load(f, mmap_mode="r") for f in files[role]]
        return state[role], state["sd"]

    def bg_snapshot(role, rng):
        """파일들을 이어 놓은 것처럼 시작 위치를 하나 뽑는다(20 ms 가 파일 경계를 넘지 않게). 반환 (IQ, 파일 번호 × 10⁸ + 시작 표본)."""
        arrs, sd = background(role)
        lens = [len(a) - g["N"] + 1 for a in arrs]
        off = int(rng.integers(0, sum(lens))); k = 0
        while off >= lens[k]:
            off -= lens[k]; k += 1
        return (np.asarray(arrs[k][off: off + g["N"]]) / sd).astype(np.complex64), k * 10 ** 8 + off

    g["background"] = background
    g["bg_snapshot"] = bg_snapshot
    g["BG_FILES"] = files
    g["NOISE_STATE"] = state


def main(hackrf):
    src = (ROOT / hackrf).resolve() if not Path(hackrf).is_absolute() else Path(hackrf)
    info = {}
    for site, rel in RAW.items():
        out = BG / site; out.mkdir(parents=True, exist_ok=True)
        fs = sorted((src / rel / "recordings").glob("*/*.cs8"))
        assert len(fs) == 5, f"{site}: .cs8 파일 5개가 있어야 하는데 {len(fs)}개"
        for k, f in enumerate(fs, 1):
            t = out / f"0{k}.npy"
            if not t.exists():
                r = np.fromfile(f, np.int8).astype(np.float32)
                np.save(t, ((r[0::2] + 1j * r[1::2]) / 128).astype(np.complex64))
            info[f"{site}/0{k}.npy"] = f.relative_to(src).as_posix()
        print(site, "→", out.relative_to(ROOT), flush=True)
    for site in ORDER:
        fs = [BG / f for f in SITES[site]["files"]["train"]]
        if not all(f.exists() for f in fs):                      # 공원 파일(Park02_*.npy)을 아직 안 받았으면 기록만 건너뛴다
            print(site, ": 학습셋 배경 파일이 없어 잡음 세기 기록을 건너뜀 (README 의 '데이터 받기')", flush=True); continue
        p, per = noise_power(fs)
        info[f"{site} 잡음 전력(학습셋 파일별)"] = per
    json.dump(info, open(BG / "site_sources.json", "w"), ensure_ascii=False, indent=1)
    print("SITE_SETUP_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hackrf", default="data/raw/hackrf")
    main(ap.parse_args().hackrf)
