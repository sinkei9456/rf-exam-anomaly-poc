"""장소별 실험 결과 모으기 (2026-11-19). 사용: python src/site_summary.py (세 장소 평가와 site_cross.py 가 끝난 뒤)

읽는 것: runs/site/<장소>/{results.json, light.json, pixel_select.json, cae<크기>_select.json}, runs/site/cross.json
출력: runs/site/results.md (표), doc/fig/site/fig_sites_background.png, doc/fig/site/fig_recall_vs_snr.png
CAE 는 시드 3개의 평균이고, 괄호 안은 최소–최대다.
"""
import sys, json
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C
from site_setup import ORDER, SITES
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evaluate as E

assert not C.SITE, "EXAM_SITE 없이 돌린다"
RUN = {s: C.ROOT / "runs" / "site" / s for s in ORDER}
FIG = C.ROOT / "doc" / "fig" / "site"
SIZES = ["65k", "21k", "6k"]
SNRS = [0, 3, 6, 9, 12]
METHODS = ["A", "B"] + [f"cae{z}" for z in SIZES]
NAME = {"A": "(가) 직전 스냅샷과의 픽셀 차이", "B": "(나) 정상 평균과의 픽셀 차이",
        "cae65k": "CAE, 파라미터 65,265개", "cae21k": "CAE, 파라미터 21,121개", "cae6k": "CAE, 파라미터 6,489개"}
SN = {s: SITES[s]["name"] for s in ORDER}
R = {s: json.load(open(RUN[s] / "results.json")) for s in ORDER}
L = {s: json.load(open(RUN[s] / "light.json")) for s in ORDER}


def vals(site, m, f):
    """방법 m 의 값 목록(픽셀 차이는 1개, CAE 는 시드 3개)."""
    keys = [m] if m in ("A", "B") else sorted(k for k in R[site] if k.startswith(m + "_s"))
    return np.array([f(R[site][k]) for k in keys])


def cell(site, m, f, scale=100, nd=1):
    v = vals(site, m, f) * scale
    if len(v) == 1: return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f} ({v.min():.{nd}f}–{v.max():.{nd}f})"


def write_md():
    T = ["# 장소별 실험 결과 (src/site_summary.py 가 만든 표)", "",
         "장소마다 그 장소 학습셋 정상으로 학습하고, threshold·점수 상자 크기·CAE 에포크는 그 장소 검증셋으로만 정했다. "
         "CAE 는 시드 3개 평균이고 괄호 안은 최소–최대다.", ""]
    T += ["## 1. 장소", "", "| 장소 | IQ 데이터 | 중심 주파수 (MHz) |", "|---|---|---:|"]
    for s in ORDER: T.append(f"| {SN[s]} | {SITES[s]['session']} | {SITES[s]['fc'] / 1e6:,.1f} |")
    T += ["", "## 2. 검증셋으로 고른 설정", "",
          "| 장소 | 방법 | 점수 상자 크기 | 고른 에포크 | threshold, 검증셋 정상의 1 %가 넘는 값 | threshold, 검증셋 정상의 0.1 %가 넘는 값 |",
          "|---|---|---|---|---:|---:|"]
    for s in ORDER:
        ps = json.load(open(RUN[s] / "pixel_select.json"))
        for m in ("A", "B"):
            T.append(f"| {SN[s]} | {NAME[m]} | {ps[m]['rows']}줄 × {ps[m]['cols']}칸 | 학습 없음 | {ps[m]['thr_fpr1']:.3f} | {ps[m]['thr_fpr01']:.3f} |")
        for z in SIZES:
            sel = json.load(open(RUN[s] / f"cae{z}_select.json"))
            for sd in sorted(sel):
                v = sel[sd]
                T.append(f"| {SN[s]} | {NAME['cae' + z]}, 시드 {sd} | {v['rows']}줄 × {v['cols']}칸 | {v['best_epoch']} | {v['thr_fpr1']:.4f} | {v['thr_fpr01']:.4f} |")
    sec = 3
    for fpr in ("0.001", "0.01"):
        p = float(fpr) * 100
        T += ["", f"## {sec}. threshold를 검증셋 정상의 {p:g} %가 넘게 정했을 때", ""]; sec += 1
        T += [f"**테스트셋 정상 스트림 20,000장에서 잰 FPR (%)**", "", "| 방법 | " + " | ".join(SN[s] for s in ORDER) + " |", "|---|" + "---:|" * len(ORDER)]
        for m in METHODS: T.append(f"| {NAME[m]} | " + " | ".join(cell(s, m, lambda r: r[fpr]["test_fpr"], 100, 2) for s in ORDER) + " |")
        T += ["", "**정상 스트림 1시간당 경보 수, 최근 10장 중 2장 이상일 때 경보 (회)**", "", "| 방법 | " + " | ".join(SN[s] for s in ORDER) + " |", "|---|" + "---:|" * len(ORDER)]
        for m in METHODS: T.append(f"| {NAME[m]} | " + " | ".join(cell(s, m, lambda r: r[fpr]["alarms_h"]["2"], 1, 1) for s in ORDER) + " |")
        for kind in (1, 2):
            for s in ORDER:
                T += ["", f"**{SN[s]}, 테스트 이상 {kind}({C.ANOM[kind]['name']})의 스냅샷 recall (%)**", "",
                      "| 방법 | " + " | ".join(f"SNR {x} dB" for x in SNRS) + " |", "|---|" + "---:|" * len(SNRS)]
                for m in METHODS: T.append(f"| {NAME[m]} | " + " | ".join(cell(s, m, lambda r, x=x: r[fpr]["cells"][f"{kind}_{x}"]["recall"]) for x in SNRS) + " |")
        for kind in (1, 2):
            for s in ORDER:
                T += ["", f"**{SN[s]}, 테스트 이상 {kind}의 이벤트 recall, 10초 중 2장 이상 threshold를 넘으면 잡은 것으로 봄 (%)**", "",
                      "| 방법 | " + " | ".join(f"SNR {x} dB" for x in SNRS) + " |", "|---|" + "---:|" * len(SNRS)]
                for m in METHODS: T.append(f"| {NAME[m]} | " + " | ".join(cell(s, m, lambda r, x=x: r[fpr]["cells"][f"{kind}_{x}"]["event_det"]["2"]) for x in SNRS) + " |")
    T += ["", f"## {sec}. 가벼운 정도", "", "파라미터 수와 모델 파일 크기는 장소와 상관없이 같다. 학습 시간과 CPU 시간은 세 장소의 평균이다.", "",
          "| 방법 | 학습 파라미터 (개) | 모델 파일 크기 (KB) | GPU 학습 시간, 20 에포크 (초) | CPU 코어 1개에서 스냅샷 1장 점수 계산 (ms) |", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        v = [L[s][m] for s in ORDER]
        if m in ("A", "B"):
            T.append(f"| {NAME[m]} | 0 | — | — | {np.mean([x['cpu_ms'] for x in v]):.1f} |")
        else:
            T.append(f"| {NAME[m]} | {v[0]['n_params']:,} | {v[0]['file_bytes'] / 1024:.0f} | {np.mean([x['train_seconds'] for x in v]):.0f} | {np.mean([x['cpu_ms'] for x in v]):.1f} |")
    sec += 1
    X = json.load(open(C.ROOT / "runs" / "site" / "cross.json"))
    T += ["", f"## {sec}. 다른 장소에서 학습한 모델을 쓸 때의 FPR", "",
          "행은 모델을 학습한 장소, 열은 시험한 장소다. 대각선 칸이 그 장소에서 학습한 경우다. threshold는 학습한 장소의 검증셋 정상의 0.1 %가 넘게 정한 값이다. CAE는 시드 3개 평균이다."]
    for m in METHODS:
        T += ["", f"**{NAME[m]}: 테스트셋 정상 스트림 20,000장에서 잰 FPR (%)**", "", "| 모델을 학습한 장소 | " + " | ".join(f"시험한 장소: {SN[y]}" for y in ORDER) + " |", "|---|" + "---:|" * len(ORDER)]
        for x in ORDER: T.append(f"| {SN[x]} | " + " | ".join(f"{X[m][x][y]['0.001'] * 100:.2f}" for y in ORDER) + " |")
    (C.ROOT / "runs" / "site" / "results.md").write_text("\n".join(T) + "\n", encoding="utf-8")


def fig_recall():
    """장소 3 × 이상 종류 2. threshold 는 검증셋 정상의 0.1 %가 넘는 값. CAE 는 시드 평균(띠 = 최소–최대)."""
    col = {"A": "#1f77b4", "B": "#ff7f0e", "cae65k": "#6a3d9a", "cae21k": "#9467bd", "cae6k": "#c5b0d5"}
    mk = {"A": "o", "B": "o", "cae65k": "^", "cae21k": "s", "cae6k": "D"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    for i, kind in enumerate((1, 2)):
        for j, s in enumerate(ORDER):
            ax = axs[i, j]
            for m in METHODS:
                v = np.array([vals(s, m, lambda r, x=x: r["0.001"]["cells"][f"{kind}_{x}"]["recall"]) for x in SNRS])   # [SNR, 시드]
                ax.plot(SNRS, v.mean(1), marker=mk[m], color=col[m], label=NAME[m])
                if v.shape[1] > 1: ax.fill_between(SNRS, v.min(1), v.max(1), color=col[m], alpha=0.15)
            ax.set_title(f"{SN[s]}: 테스트 이상 {kind} ({C.ANOM[kind]['name']})", fontsize=10)
            ax.set_xlabel("이상 신호 SNR (dB)"); ax.set_ylabel("스냅샷 recall"); ax.set_ylim(-0.02, 1.05); ax.set_xticks(SNRS); ax.grid(alpha=0.3)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=9)
    fig.savefig(FIG / "fig_recall_vs_snr.png", dpi=120); plt.close(fig)


def fig_sites():
    """장소마다 받은 IQ 데이터 그대로의 스냅샷 1장(학습셋 첫 파일의 처음 20 ms). 잡음 세기는 그 장소 학습셋 파일로 맞춘 값."""
    from site_setup import noise_power, BG
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, s in zip(axs, ORDER):
        fs = [BG / f for f in SITES[s]["files"]["train"]]; p, _ = noise_power(fs)
        x = np.asarray(np.load(fs[0], mmap_mode="r")[:C.N]) / np.sqrt(p)
        fc = SITES[s]["fc"]; ext = [E.EXT[0], E.EXT[1], (fc - C.FS / 2) / 1e6, (fc + C.FS / 2) / 1e6]
        im = ax.imshow(C.spec_db(x), origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=-3, vmax=20, interpolation="nearest")
        ax.set_title(f"{SN[s]}: 학습셋 파일 {fs[0].stem}, 0–20 ms", fontsize=10); ax.set_xlabel("시간 (ms)"); ax.set_ylabel("주파수 (MHz)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="배경 잡음 세기 대비 밝기 (dB)")
    fig.savefig(FIG / "fig_sites_background.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    write_md(); fig_recall(); fig_sites()
    print("SITE_SUMMARY_DONE", flush=True)
