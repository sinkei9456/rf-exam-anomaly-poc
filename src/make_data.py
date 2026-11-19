"""학습셋·검증셋·테스트셋 스냅샷 만들기 (2026-11-18). CPU 만 쓴다.

스냅샷은 1초에 한 장씩(20 ms) 받는다고 가정한다. 스트림 = 시간 순서대로 이어진 스냅샷.
| 데이터셋 이름        | 배경 노이즈        | 내용 |
| train_normal  10,000 | 공원 무신호 0–1초 | 정상(배경 + 아주 작은 와이파이·블루투스) |
| train_blob    10,000 | 0–1초             | 정상 + 모양을 정하지 않은 사각형 덩어리(CNN 학습용 이상) |
| val_normal     5,000 | 1–2초             | 정상 스트림(연속). threshold 는 여기서만 정한다 |
| val_blob       2,000 | 1–2초             | 정상 + 사각형 덩어리. 평활 크기·CNN 에포크 선택 |
| test_normal   20,000 | 2–3초             | 정상 스트림(연속, 1초 간격이면 5시간 33분) |
| test_event     4,400 | 2–3초             | 이상 이벤트: 종류 2 × SNR 5단계(0·3·6·9·12 dB) × 40개 × (앞 정상 1장 + 기기 켜짐 10장) |
| test_demo      3,600 | 2–3초             | 1시간 시연 스트림, 이상 이벤트 3번(각 10초)만 들어감 |
테스트용 이상(블루투스 2종)은 학습셋·검증셋 어디에도 없다.
출력: data/<이름>.npy (uint8, 장수 × 256 × 97), data/<이름>.csv (스냅샷마다 정보), data/<이름>_boxes.json (넣은 신호 박스)
장소별 실험: EXAM_SITE=<장소> python src/make_data.py → data/site/<장소>/ (train_blob·test_demo 는 만들지 않음, site_setup.py 참고)
사용: python src/make_data.py [--workers 16]
"""
import sys, json, time, argparse
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C

ROLE = {"train": 1, "val": 2, "test": 3}
KIND = {"normal": 1, "blob": 2, "event": 3, "demo": 4}
SNRS = [0, 3, 6, 9, 12]
N_EVENT, EV_LEN = 40, 10
DEMO_LEN = 3600
DEMO_EVENTS = [(700, 1, 6.0), (1900, 2, 6.0), (3000, 1, 3.0)]        # (시작 위치, 이상 종류, SNR dB), 각 10장


def snapshot(spec):
    role, kind, idx, anom, snr = spec["role"], spec["kind"], spec["idx"], spec.get("anom", 0), spec.get("snr", np.nan)
    rng = np.random.default_rng([ROLE[role], KIND[kind], idx])
    x, off = C.bg_snapshot(role, rng)
    boxes = C.add_normal(x, rng)
    n_tx = n_vis = 0
    if kind == "blob":
        boxes += C.add_blobs(x, rng)
    if anom:
        b, n_tx, n_vis = C.add_anomaly(x, rng, anom, snr); boxes += b
    info = dict(idx=idx, role=role, kind=kind, anom=anom, snr=snr, event=spec.get("event", -1), pos=spec.get("pos", -1),
                n_tx=n_tx, n_vis=n_vis, n_wifi=sum(b[0] == "wifi" for b in boxes), n_bt=sum(b[0] == "bt" for b in boxes),
                n_blob=sum(b[0] == "blob" for b in boxes), bg_offset=off)
    return C.to_u8(C.spec_db(x)), info, [[b[0]] + [round(float(v), 7) for v in b[1:]] for b in boxes]


def specs():
    S = {}
    S["train_normal"] = [dict(role="train", kind="normal", idx=i) for i in range(10000)]
    S["train_blob"] = [dict(role="train", kind="blob", idx=i) for i in range(10000)]
    S["val_normal"] = [dict(role="val", kind="normal", idx=i) for i in range(5000)]
    S["val_blob"] = [dict(role="val", kind="blob", idx=i) for i in range(2000)]
    S["test_normal"] = [dict(role="test", kind="normal", idx=i) for i in range(20000)]
    ev, i, e = [], 0, 0
    for a in (1, 2):
        for snr in SNRS:
            for _ in range(N_EVENT):
                for pos in range(EV_LEN + 1):                    # pos 0 = 이벤트 바로 앞 정상 스냅샷(직전 스냅샷과의 픽셀차용)
                    ev.append(dict(role="test", kind="event", idx=i, anom=a if pos else 0, snr=float(snr) if pos else np.nan, event=e, pos=pos)); i += 1
                e += 1
    S["test_event"] = ev
    demo = []
    for t in range(DEMO_LEN):
        sp = dict(role="test", kind="demo", idx=t)
        for k, (t0, a, snr) in enumerate(DEMO_EVENTS):
            if t0 <= t < t0 + EV_LEN: sp.update(anom=a, snr=snr, event=k, pos=t - t0 + 1)
        demo.append(sp)
    S["test_demo"] = demo
    return S


def main(workers):
    C.DATA.mkdir(parents=True, exist_ok=True)
    for r in ("train", "val", "test"): C.background(r)                    # 부모에서 rms 를 미리 계산(자식이 물려받음)
    with Pool(workers) as P:
        for name, sp in specs().items():
            if C.SITE and name in ("train_blob", "test_demo"): continue      # 장소별 실험(EXAM_SITE)에는 CNN 학습용 사각형·1시간 시연 스트림을 만들지 않는다
            if (C.DATA / f"{name}.npy").exists(): print("있음, 건너뜀:", name, flush=True); continue
            t = time.time(); out = P.map(snapshot, sp, chunksize=32)
            imgs = np.stack([o[0] for o in out]); info = [o[1] for o in out]; boxes = [o[2] for o in out]
            np.save(C.DATA / f"{name}.tmp.npy", imgs); (C.DATA / f"{name}.tmp.npy").rename(C.DATA / f"{name}.npy")
            keys = list(info[0].keys())
            (C.DATA / f"{name}.csv").write_text(",".join(keys) + "\n" + "".join(",".join(str(r[k]) for k in keys) + "\n" for r in info))
            json.dump(boxes, open(C.DATA / f"{name}_boxes.json", "w"))
            print(f"{name}: {len(imgs)}장, {time.time() - t:.0f}초", flush=True)
    print("MAKE_DATA_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); main(ap.parse_args().workers)
