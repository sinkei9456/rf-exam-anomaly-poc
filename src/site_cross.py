"""장소 바꿔 쓰기 (2026-11-19). 한 장소에서 학습한 모델을 다른 장소의 테스트셋 정상 스트림 20,000장에 그대로 써서 FPR 을 잰다.
장소마다 따로 학습해야 하는지 확인하는 용도다. 사용: python src/site_cross.py (세 장소의 학습·평가가 끝난 뒤, GPU 1개)

모델 장소 X → 데이터 장소 Y:
  - threshold, 점수 상자 크기, (나)의 픽셀별 정상 평균·표준편차, CAE 가중치와 입력 표준화 값은 모두 X 에서 정한 것을 그대로 쓴다.
  - 데이터의 잡음 세기는 Y 의 학습셋 파일로 맞춘 값이다(새 장소에서도 배경 잡음 세기만은 잰다고 가정). 장소마다 수신 이득은 같지만
    공원 배경 파일은 11-18 에 크기를 바꿔 저장해 두어서, 받은 값 그대로는 장소끼리 비교할 수 없다.
  - X = Y 칸은 site_evaluate.py 의 테스트셋 FPR 과 같다.
출력: runs/site/cross.json  {모델: {X: {Y: {"0.01": FPR, "0.001": FPR}}}}  (CAE 는 시드 3개 평균)
"""
import sys, json, gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from site_setup import ORDER
from pixeldiff import score_maps
from cae import CAE, SIZES, scores_all
from cnn import Prep, dev

assert not C.SITE, "EXAM_SITE 없이 돌린다(세 장소를 모두 읽음)"
ROOT = C.ROOT
RUN = {s: ROOT / "runs" / "site" / s for s in ORDER}
DAT = {s: ROOT / "data" / "site" / s for s in ORDER}


def fpr(scores, sel):
    v = scores[~np.isnan(scores)]
    return {"0.01": float((v > sel["thr_fpr1"]).mean()), "0.001": float((v > sel["thr_fpr01"]).mean())}


def main():
    U8 = {y: np.load(DAT[y] / "test_normal.npy") for y in ORDER}
    DB = {y: C.to_db(U8[y]) for y in ORDER}
    out = {"A": {}, "B": {}, **{f"cae{z}": {} for z in SIZES}}
    for x in ORDER:
        ps = json.load(open(RUN[x] / "pixel_select.json")); st = np.load(RUN[x] / "pixel_stats.npz")
        prep = Prep.__new__(Prep); prep.mu = torch.tensor(st["mu"], device=dev); prep.sd = torch.tensor(st["sd"], device=dev)
        for m in ("A", "B"): out[m][x] = {}
        for z in SIZES: out[f"cae{z}"][x] = {}
        for y in ORDER:
            D = DB[y]
            A = np.concatenate([D[:1], D[1:] - D[:-1]]); A[0] = 0
            s, _ = score_maps(A, ps["A"]["rows"], ps["A"]["cols"]); s[0] = np.nan; out["A"][x][y] = fpr(s, ps["A"])
            s, _ = score_maps((D - st["mu"]) / st["sd"], ps["B"]["rows"], ps["B"]["cols"]); out["B"][x][y] = fpr(s, ps["B"])
            Xg = torch.from_numpy(U8[y]).to(dev)
            for z in SIZES:
                sel = json.load(open(RUN[x] / f"cae{z}_select.json")); v = []
                for sd in sorted(sel):
                    net = CAE(tuple(SIZES[z])).to(dev); net.load_state_dict(torch.load(RUN[x] / f"cae{z}_s{sd}.pt", map_location=dev))
                    sc = scores_all(net, prep, Xg, [(sel[sd]["rows"], sel[sd]["cols"])])[0]; v.append(fpr(sc, sel[sd]))
                    del net
                out[f"cae{z}"][x][y] = {k: float(np.mean([w[k] for w in v])) for k in ("0.01", "0.001")}
            del Xg; gc.collect(); torch.cuda.empty_cache()
            print(x, "→", y, {m: out[m][x][y]["0.001"] for m in out}, flush=True)
    json.dump(out, open(ROOT / "runs" / "site" / "cross.json", "w"), ensure_ascii=False, indent=1)
    print("SITE_CROSS_DONE", flush=True)


if __name__ == "__main__":
    main()
