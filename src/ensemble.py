"""(마) CNN·CAE 앙상블 (2026-11-18). 학습은 하지 않는다. 이미 계산한 CNN·CAE 점수를 합친다. CPU.

규칙(테스트셋을 보기 전에 정함):
  1. 눈금 맞추기: CNN 점수와 CAE 점수는 단위가 달라 그대로 비교할 수 없다. 모델마다 검증셋 정상 스냅샷 5,000장의 점수로
     (점수 - 중앙값) / (검증셋 정상의 1 %가 넘는 값 - 중앙값) 으로 바꾼다. 두 모델 모두 정상 중앙값이 0, 정상의 1 %가 넘는 값이 1 이 된다.
  2. 앙상블 점수 = 눈금 맞춘 두 값 중 큰 값. 둘 중 하나라도 정상에서 드물다고 보면 이상으로 본다.
  3. threshold 두 가지: 검증셋 정상 스냅샷의 앙상블 점수 중 1 %가 넘는 값, 0.1 %가 넘는 값.
  4. CNN 시드 k 와 CAE 시드 k 를 짝지어 앙상블 3개(k = 0, 1, 2)를 만든다.
가중치를 고르지 않는 이유: 검증셋의 이상 데이터(사각형 덩어리)는 CNN 이 학습한 것과 같은 종류라, 그것으로 가중치를 고르면 CNN 쪽으로 쏠린다.
그래서 결합 규칙은 미리 정하고, 검증셋은 정상 스냅샷으로 눈금과 threshold 를 맞추는 데만 쓴다. 검증셋 사각형 덩어리 recall 은 참고로만 적는다.
입력: runs/scores_cnn.npz(키 "C<시드>/<데이터셋>"), runs/scores_cae.npz(키 "E<시드>/<데이터셋>")
출력: runs/scores_ens.npz(키 "N<시드>/<데이터셋>"), runs/ens_select.json(시드마다 눈금 맞춤 값, threshold, 참고용 검증 recall)
사용: python src/ensemble.py
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SETS = ["val_normal", "val_blob", "test_normal", "test_event", "test_demo"]


def fit_scale(val_normal_scores):
    """검증셋 정상 점수의 중앙값과 1 %가 넘는 값(상위 1 %)."""
    v = np.asarray(val_normal_scores, dtype=np.float64)
    return float(np.median(v)), float(np.quantile(v, 0.99))


def normalize(s, med, q99):
    return (np.asarray(s, dtype=np.float64) - med) / (q99 - med)


def combine(s_cnn, s_cae, p):
    """p = 눈금 맞춤 값 dict(cnn_med, cnn_q99, cae_med, cae_q99). 반환: 앙상블 점수."""
    return np.maximum(normalize(s_cnn, p["cnn_med"], p["cnn_q99"]), normalize(s_cae, p["cae_med"], p["cae_q99"]))


def main():
    Q = np.load(C.RUNS / "scores_cnn.npz"); E = np.load(C.RUNS / "scores_cae.npz")
    out, sel = {}, {}
    for s in ("0", "1", "2"):
        cm, cq = fit_scale(Q[f"C{s}/val_normal"]); em, eq = fit_scale(E[f"E{s}/val_normal"])
        p = dict(cnn_med=cm, cnn_q99=cq, cae_med=em, cae_q99=eq)
        for n in SETS: out[f"N{s}/{n}"] = combine(Q[f"C{s}/{n}"], E[f"E{s}/{n}"], p)
        v, b = out[f"N{s}/val_normal"], out[f"N{s}/val_blob"]
        t1, t01 = float(np.quantile(v, 0.99)), float(np.quantile(v, 0.999))
        sel[s] = dict(**p, thr_fpr1=t1, thr_fpr01=t01, val_blob_recall_thr1=float((b > t1).mean()), val_blob_recall_thr01=float((b > t01).mean()))
        print(f"앙상블 시드 {s}: CNN 눈금(중앙값 {cm:.3f}, 1 % 값 {cq:.3f}), CAE 눈금(중앙값 {em:.4f}, 1 % 값 {eq:.4f}), "
              f"threshold(검증셋 정상의 1 %가 넘는 값) {t1:.4f}, threshold(0.1 %가 넘는 값) {t01:.4f}, "
              f"참고: 검증셋 사각형 덩어리 recall {sel[s]['val_blob_recall_thr1'] * 100:.1f} %(1 % threshold)", flush=True)
    np.savez(C.RUNS / "scores_ens.npz", **out)
    json.dump(sel, open(C.RUNS / "ens_select.json", "w"), ensure_ascii=False, indent=1)
    print("ENSEMBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
