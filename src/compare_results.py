"""내 실행 결과(runs/results.json)를 공개한 기준 결과(results/results.json)와 비교한다 (2026-11-18).

방법마다 테스트셋 FPR, 스냅샷 recall, 이벤트 recall, AUROC 의 차이(퍼센트 포인트, AUROC 는 그대로)를 보여 준다.
학습이 없는 (가)(나)는 차이가 0 이어야 한다. CNN·CAE 를 저장소의 학습된 모델(models/)로 추론만 했으면(run_all.sh --use-released-models) 0 에 가깝고,
처음부터 다시 학습했으면 GPU 계산 순서 차이 때문에 조금 다르다.
사용: python src/compare_results.py [내 결과.json] [기준 결과.json]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = {"A": "(가) 직전 스냅샷과의 픽셀차", "B": "(나) 정상 평균과의 픽셀차", "C": "(다) CNN", "E": "(라) CAE"}


def flat(r):
    """결과 한 방법 → {지표 이름: 값}."""
    out = {}
    cell = lambda k: "이상 {}, SNR {} dB".format(*k.split("_"))
    for fpr in ("0.01", "0.001"):
        t = f"threshold: 검증셋 정상의 {float(fpr) * 100:g} %가 넘는 값"
        out[f"테스트셋 FPR ({t})"] = r[fpr]["test_fpr"]
        for k, c in r[fpr]["cells"].items():
            out[f"{cell(k)} recall ({t})"] = c["recall"]; out[f"{cell(k)} 이벤트 recall ({t})"] = c["event_det"]["2"]
    for k, v in r["auroc"].items(): out[f"{cell(k)} AUROC"] = v
    return out


def main(mine, ref):
    A, B = json.load(open(mine, encoding="utf-8")), json.load(open(ref, encoding="utf-8"))
    worst = 0.0
    print(f"비교: {mine}  대  {ref}")
    print("| 방법 | 지표 수 | 다른 지표 수 | 가장 큰 차이 | 그 지표 |")
    print("|---|---:|---:|---:|---|")
    for m in sorted(B):
        if m not in A: print(f"| {m} | — | 내 결과에 없음 | | |"); continue
        a, b = flat(A[m]), flat(B[m]); d = {k: abs(a[k] - b[k]) for k in b}
        k = max(d, key=d.get); n_diff = sum(v > 1e-6 for v in d.values())
        scale = 1 if k.endswith("AUROC") else 100
        unit = "" if k.endswith("AUROC") else " %p"
        name = NAME.get(m[0], m) + (f" 시드 {m[1:]}" if m[1:].isdigit() else "")
        print(f"| {name} | {len(d)} | {n_diff} | {d[k] * scale:.3f}{unit} | {k} |")
        worst = max(worst, d[k] * scale)
    print(f"가장 큰 차이 {worst:.3f}" + (" — 기준 결과와 같음" if worst < 1e-6 else ""))


if __name__ == "__main__":
    mine = sys.argv[1] if len(sys.argv) > 1 else str(C.RUNS / "results.json")
    ref = sys.argv[2] if len(sys.argv) > 2 else str(C.RESULTS / "results.json")
    main(mine, ref)
