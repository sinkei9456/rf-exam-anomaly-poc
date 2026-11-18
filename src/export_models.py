"""학습 결과(runs/)를 추론용 모델 폴더(models/)로 내보낸다 (2026-11-18).

복사: runs/cnn_s{0,1,2}.pt, runs/cae_s{0,1,2}.pt, runs/pixel_stats.npz → models/
만듦: models/thresholds.json
  pixel: (나) 정상 평균과의 픽셀차의 평활 크기(rows 줄 × cols 칸)와 threshold 두 가지(검증셋 정상의 1 %·0.1 %가 넘는 값)
  cnn:   시드마다 threshold, 고른 에포크, 에포크별 검증 기록
  cae:   시드마다 평활 크기, threshold, 고른 에포크, 에포크별 검증 기록
저장소의 models/ 는 2026-11-18 학습 결과를 이 스크립트로 내보낸 것이다.
사용: python src/export_models.py   (다시 학습한 모델로 models/ 를 바꿀 때)
"""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    C.MODELS.mkdir(exist_ok=True)
    for f in ["pixel_stats.npz"] + [f"{m}_s{s}.pt" for m in ("cnn", "cae") for s in (0, 1, 2)]:
        shutil.copy2(C.RUNS / f, C.MODELS / f)
    ps = json.load(open(C.RUNS / "pixel_select.json", encoding="utf-8"))["B"]
    cs = json.load(open(C.RUNS / "cnn_select.json", encoding="utf-8"))
    es = json.load(open(C.RUNS / "cae_select.json", encoding="utf-8"))
    meta = {
        "input": dict(fs_hz=C.FS, snapshot_s=C.WIN, center_hz=C.FC, image="uint8 256 x 97, 0 = -6 dB, 255 = +34 dB, noise floor = 0 dB"),
        "pixel": dict(rows=ps["rows"], cols=ps["cols"], thr={"0.01": ps["thr_fpr1"], "0.001": ps["thr_fpr01"]}),
        "cnn": {s: dict(thr={"0.01": v["thr_fpr1"], "0.001": v["thr_fpr01"]}, best_epoch=v["best_epoch"], history=v["history"]) for s, v in cs.items()},
        "cae": {s: dict(rows=v["rows"], cols=v["cols"], thr={"0.01": v["thr_fpr1"], "0.001": v["thr_fpr01"]},
                        best_epoch=v["best_epoch"], history=v["history"]) for s, v in es.items()},
    }
    json.dump(meta, open(C.MODELS / "thresholds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("models/ 에 내보냄:", sorted(p.name for p in C.MODELS.iterdir()))


if __name__ == "__main__":
    main()
