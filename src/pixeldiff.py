"""픽셀차 이상 탐지 (2026-11-18). 학습 없음(정상 평균·표준편차 계산만), CPU.

(가) 직전 스냅샷과의 픽셀차: d = 지금 스냅샷(dB) - 1초 전 스냅샷(dB).
(나) 정상 평균과의 픽셀차: z = (지금 스냅샷 - 픽셀별 정상 평균) / 픽셀별 정상 표준편차. 평균·표준편차는 학습셋 정상 10,000장으로 계산.
두 방법 모두 d 또는 z 를 r줄 × c칸 상자 안에서 평균(평활)한 뒤, 이미지 전체에서 가장 큰 값을 이상 점수로 쓴다.
  - r × c = 1 × 1 이면 평활 없이 픽셀 하나의 최댓값이다.
  - 후보 r = 1·6·13·26줄(0.08·0.5·1·2 MHz), c = 1·2·5·10·15칸(0.2·0.4·1·2·3 ms).
선택 규칙(테스트셋을 보지 않음): 검증셋 정상 스트림 점수의 상위 1 % 값을 threshold 로 두었을 때, 검증셋 사각형 덩어리 2,000장 중 threshold 를 넘는 비율(recall)이 가장 큰 크기.
threshold 두 가지: 검증셋 정상 스트림 스냅샷의 1 %가 넘는 값과 0.1 %가 넘는 값.
출력: runs/pixel_stats.npz(평균·표준편차), runs/pixel_select.json(후보별 검증 결과, 고른 크기, threshold), runs/scores_pixel.npz(데이터셋별 점수, 점수 위치)
"""
import sys, json
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C

ROWS, COLS = [1, 6, 13, 26], [1, 2, 5, 10, 15]
SETS = ["val_normal", "val_blob", "test_normal", "test_event"] + ([] if C.SITE else ["test_demo"])   # 장소별 실험에는 1시간 시연 스트림이 없다


def load(name):
    return C.to_db(np.load(C.DATA / f"{name}.npy"))


def prev_of(name, X, ref=None):
    """직전 스냅샷. 스트림은 바로 앞 장, 첫 장은 자기 자신(점수 0). val_blob 은 같은 번호의 검증 정상 스냅샷, test_event 는 이벤트 안 바로 앞 장."""
    if name == "val_blob": return ref[: len(X)]
    P = np.concatenate([X[:1], X[:-1]])
    return P


def valid(r, c):
    """상자가 이미지 안에 완전히 들어가는 가운데 위치(줄 범위, 칸 범위). 2026-11-18 고침: 처음에는 가장자리 밖을 가장자리 값으로 채운 상자(mode=nearest)까지
    최댓값 후보로 봐서, 가장자리에 걸린 짧은 신호가 상자 평균에 여러 번 들어가 점수가 부풀었다(검증셋 정상의 1 %가 넘게 정한 threshold 를 넘은 검증셋 정상 스냅샷의 88 %가 앞뒤 2칸)."""
    return slice(r // 2, C.NF - (r - 1 - r // 2)), slice(c // 2, C.NT - (c - 1 - c // 2))


def smooth(M, r, c):
    """r줄 × c칸 상자 평균. 상자가 이미지 밖으로 나가는 위치는 NaN."""
    F = ndimage.uniform_filter(M, size=(1, r, c) if M.ndim == 3 else (r, c), mode="nearest")
    out = np.full_like(F, np.nan); vr, vc = valid(r, c); out[..., vr, vc] = F[..., vr, vc]; return out


def score_maps(M, r, c, bs=1000):
    """평활한 지도에서 상자가 이미지 안에 완전히 들어가는 위치만 보고, 그중 최댓값과 그 위치(줄, 칸)."""
    s = np.empty(len(M), np.float32); loc = np.empty((len(M), 2), np.int32); vr, vc = valid(r, c)
    for i in range(0, len(M), bs):
        F = ndimage.uniform_filter(M[i:i + bs], size=(1, r, c), mode="nearest")[:, vr, vc]
        flat = F.reshape(len(F), -1); k = flat.argmax(1)
        s[i:i + bs] = flat[np.arange(len(F)), k]; loc[i:i + bs] = np.stack(np.unravel_index(k, F.shape[1:]), 1) + [vr.start, vc.start]
    return s, loc


def thr(scores, fpr):
    return float(np.quantile(scores, 1 - fpr))


def main():
    C.RUNS.mkdir(parents=True, exist_ok=True)
    Xn = load("train_normal"); mu = Xn.mean(0); sd = np.maximum(Xn.std(0), 0.3); del Xn
    np.savez(C.RUNS / "pixel_stats.npz", mu=mu, sd=sd)
    D = {n: load(n) for n in SETS}
    maps = {}
    for n in SETS:
        maps[("B", n)] = (D[n] - mu) / sd
        A = D[n] - prev_of(n, D[n], D["val_normal"])
        if n in ("test_normal", "test_demo", "val_normal"): A[0] = 0
        maps[("A", n)] = A
    sel = {}
    for m in ("A", "B"):
        rows = []
        for r in ROWS:
            for c in COLS:
                sn, _ = score_maps(maps[(m, "val_normal")][1:], r, c); sb, _ = score_maps(maps[(m, "val_blob")], r, c)
                t1 = thr(sn, 0.01); rows.append(dict(rows=r, cols=c, thr_fpr1=t1, val_recall_fpr1=float((sb > t1).mean()),
                                                      val_auroc=float((sb[:, None] > sn[None, :]).mean())))
                print(m, r, c, f"검증셋 사각형 덩어리 recall(threshold는 검증셋 정상의 1 %가 넘는 값) {rows[-1]['val_recall_fpr1']:.3f}  AUROC {rows[-1]['val_auroc']:.4f}", flush=True)
        best = max(rows, key=lambda d: (d["val_recall_fpr1"], d["val_auroc"]))
        sel[m] = dict(candidates=rows, rows=best["rows"], cols=best["cols"])
    out = {}
    for m in ("A", "B"):
        r, c = sel[m]["rows"], sel[m]["cols"]
        for n in SETS:
            s, loc = score_maps(maps[(m, n)], r, c)
            if m == "A" and n in ("test_normal", "test_demo", "val_normal"): s[0] = np.nan
            out[f"{m}/{n}"] = s; out[f"{m}/{n}/loc"] = loc
        sn = out[f"{m}/val_normal"][1:]
        sel[m].update(thr_fpr1=thr(sn, 0.01), thr_fpr01=thr(sn, 0.001))
        print(m, f"고른 크기 {r}줄 × {c}칸, threshold(검증셋 정상의 1 %가 넘는 값) {sel[m]['thr_fpr1']:.4f}, threshold(0.1 %가 넘는 값) {sel[m]['thr_fpr01']:.4f}", flush=True)
    np.savez(C.RUNS / "scores_pixel.npz", **out)
    json.dump(sel, open(C.RUNS / "pixel_select.json", "w"), ensure_ascii=False, indent=1)
    print("PIXELDIFF_DONE", flush=True)


if __name__ == "__main__":
    main()
