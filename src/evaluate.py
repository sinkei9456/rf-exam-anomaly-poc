"""테스트셋 평가와 그림 (2026-11-18). threshold 는 검증셋에서 정한 값만 쓴다(pixel_select.json, cnn_select.json).

스냅샷 단위
  - FPR: 테스트셋 정상 스트림 20,000장 중 threshold 를 넘은 비율.
  - recall: 테스트셋 이상 이벤트 중 이상 신호가 1개 이상 보이는 스냅샷(n_vis >= 1)에서 threshold 를 넘은 비율. 종류 2 × SNR 5단계마다.
  - AUROC: 같은 이상 스냅샷 대 테스트셋 정상 스트림.
이벤트 단위(기기가 10초 켜져 있음, 1초에 한 장)
  - 이벤트 recall: 이벤트 10장 중 k장 이상 threshold 를 넘은 비율(k = 1, 2). 아무것도 안 보이는 스냅샷도 그대로 포함.
  - 시간당 경보 수: 정상 스트림에서 '최근 10장 중 k장 이상'이 처음 참이 되는 횟수 / 5.56시간.
정밀도(희소성 반영): 이상 스냅샷 비율 p(0.1 %, 1 %)를 가정해 recall·p / (recall·p + FPR·(1 - p)).
출력: runs/results.json, runs/results.md, doc/fig/fig1–fig5 (그림 안 글자는 한국어)
"""
import sys, json
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

def _korean_font():
    """그림 안 한국어 글꼴. 나눔고딕 파일이 있으면 직접 등록하고, 없으면 설치된 한글 글꼴(나눔·Noto CJK·맑은 고딕)을 찾는다."""
    from pathlib import Path
    for f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/nanum/NanumGothic.ttf"):
        if Path(f).exists():
            fm.fontManager.addfont(f); return "NanumGothic"
    names = {x.name for x in fm.fontManager.ttflist}
    for n in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
        if n in names: return n
    print("경고: 한글 글꼴을 찾지 못해 그림의 한국어가 깨질 수 있음(나눔고딕 설치 권장)", flush=True)
    return None


_kf = _korean_font()
if _kf: plt.rcParams["font.family"] = _kf
plt.rcParams["axes.unicode_minus"] = False

SNRS = [0, 3, 6, 9, 12]
NAME = {"A": "(가) 직전 스냅샷과의 픽셀차", "B": "(나) 정상 평균과의 픽셀차", "CNN": "(다) CNN", "CAE": "(라) CAE", "ENS": "(마) CNN·CAE 앙상블"}
EXT = [(C.NFFT / 2 - C.HOP / 2) / C.FS * 1e3, ((C.NT - 1) * C.HOP + C.NFFT / 2 + C.HOP / 2) / C.FS * 1e3, (C.FC - C.FS / 2) / 1e6, (C.FC + C.FS / 2) / 1e6]


def read_csv(name):
    L = (C.DATA / f"{name}.csv").read_text().strip().split("\n"); keys = L[0].split(",")
    cols = list(zip(*[r.split(",") for r in L[1:]]))
    out = {}
    for k, v in zip(keys, cols):
        try: out[k] = np.array(v, dtype=float)
        except ValueError: out[k] = np.array(v)
    return out


LEARNED = {"CNN": ("C", "scores_cnn.npz", "cnn_select.json"), "CAE": ("E", "scores_cae.npz", "cae_select.json"),
           "ENS": ("N", "scores_ens.npz", "ens_select.json")}   # 시드 여러 개인 모델(앙상블은 CNN 시드 k + CAE 시드 k)


def load_scores():
    P = np.load(C.RUNS / "scores_pixel.npz"); ps = json.load(open(C.RUNS / "pixel_select.json"))
    M, sel = {}, {}
    for m in ("A", "B"):
        M[m] = dict(s={n: P[f"{m}/{n}"] for n in ("test_normal", "test_event", "test_demo")}, thr={0.01: ps[m]["thr_fpr1"], 0.001: ps[m]["thr_fpr01"]})
    for g, (pre, fs, fj) in LEARNED.items():
        if not (C.RUNS / fs).exists(): continue
        Q = np.load(C.RUNS / fs); sel[g] = json.load(open(C.RUNS / fj))
        for s in sorted(sel[g]):
            M[f"{pre}{s}"] = dict(s={n: Q[f"{pre}{s}/{n}"] for n in ("test_normal", "test_event", "test_demo")}, thr={0.01: sel[g][s]["thr_fpr1"], 0.001: sel[g][s]["thr_fpr01"]})
    return M, ps, sel


def alarms_per_hour(flags, k, L=10):
    w = np.convolve(flags.astype(int), np.ones(L, int))[: len(flags)]; a = w >= k
    return float((a[1:] & ~a[:-1]).sum() + a[0]) / (len(flags) / 3600)


def evaluate(M, ev):
    R = {}
    for m, d in M.items():
        sn = d["s"]["test_normal"]; ok = ~np.isnan(sn); sn_ok = sn[ok]; se = d["s"]["test_event"]
        r = dict()
        for fpr, t in d["thr"].items():
            fl = np.where(ok, sn > t, False)
            rr = dict(thr=t, test_fpr=float(fl[ok].mean()), alarms_h={k: alarms_per_hour(fl, k) for k in (1, 2)}, cells={})
            for a in (1, 2):
                for s in SNRS:
                    q = (ev["anom"] == a) & (ev["snr"] == s); vis = q & (ev["n_vis"] >= 1)
                    rec = float((se[vis] > t).mean())
                    evs = np.unique(ev["event"][q]); det = {k: float(np.mean([(se[q & (ev["event"] == e)] > t).sum() >= k for e in evs])) for k in (1, 2)}
                    prec = {p: rec * p / (rec * p + rr["test_fpr"] * (1 - p) + 1e-12) for p in (0.001, 0.01)}
                    rr["cells"][f"{a}_{s}"] = dict(n_vis=int(vis.sum()), n_all=int(q.sum()), recall=rec, event_det=det, precision=prec)
            r[fpr] = rr
        auc = {}
        for a in (1, 2):
            for s in SNRS:
                vis = (ev["anom"] == a) & (ev["snr"] == s) & (ev["n_vis"] >= 1)
                y = np.r_[np.zeros(len(sn_ok)), np.ones(vis.sum())]; auc[f"{a}_{s}"] = float(roc_auc_score(y, np.r_[sn_ok, se[vis]]))
        r["auroc"] = auc; R[m] = r
    return R


def agg(R, g, f):
    """학습 모델 g(CNN, CAE)의 시드별 값 평균·최소·최대. f(결과 dict) → 숫자."""
    pre = LEARNED[g][0]; v = np.array([f(R[m]) for m in R if m[0] == pre and m[1:].isdigit()]); return v.mean(), v.min(), v.max()


def groups(R):
    return [g for g in LEARNED if any(m[0] == LEARNED[g][0] and m[1:].isdigit() for m in R)]


def pct(x): return f"{x * 100:.1f}"


def write_md(R, ps, sel, ev):
    G = groups(R)
    L = ["# 테스트셋 결과 (src/evaluate.py 가 만든 표)", "",
         "threshold 는 모두 검증셋 정상 스트림에서 정했다. CNN·CAE 는 시드 3개의 평균이고 괄호 안은 최소–최대.", ""]
    L += ["## 1. 고른 설정 (검증셋으로 고름)", "", "| 방법 | 고른 설정 | threshold, 검증셋 정상의 1 %가 넘는 값 | threshold, 검증셋 정상의 0.1 %가 넘는 값 |", "|---|---|---:|---:|"]
    for m in ("A", "B"):
        L.append(f"| {NAME[m]} | 평활 {ps[m]['rows']}줄 × {ps[m]['cols']}칸 ({ps[m]['rows'] * C.DF_ROW / 1e6:.2f} MHz × {ps[m]['cols'] * C.DT_COL * 1e3:.1f} ms) | {ps[m]['thr_fpr1']:.2f} | {ps[m]['thr_fpr01']:.2f} |")
    for g in G:
        for s in sorted(sel[g]):
            v = sel[g][s]
            if g == "ENS":
                what = f"CNN 눈금(중앙값 {v['cnn_med']:.3f}, 1 % 값 {v['cnn_q99']:.3f}), CAE 눈금(중앙값 {v['cae_med']:.4f}, 1 % 값 {v['cae_q99']:.4f})"
            else:
                h = v["history"][v["best_epoch"] - 1]
                what = (f"에포크 {v['best_epoch']} (검증 recall {h['val_recall_fpr1'] * 100:.1f} %)" if g == "CNN" else
                        f"에포크 {v['best_epoch']} (검증 정상 복원 오차 {h['val_loss']:.4f}), 평활 {v['rows']}줄 × {v['cols']}칸")
            L.append(f"| {NAME[g]} 시드 {s} | {what} | {v['thr_fpr1']:.3f} | {v['thr_fpr01']:.3f} |")
    def rows(f, fmt=pct, rng=True):
        out = [f"| {NAME[m]} | " + " | ".join(fmt(f(R[m], s)) for s in SNRS) + " |" for m in ("A", "B")]
        for g in G:
            out.append(f"| {NAME[g]} | " + " | ".join(("{} ({}–{})".format(*map(fmt, agg(R, g, lambda r: f(r, s)))) if rng else fmt(agg(R, g, lambda r: f(r, s))[0])) for s in SNRS) + " |")
        return out
    hdr = ["| 방법 | " + " | ".join(f"SNR {s} dB" for s in SNRS) + " |", "|---|" + "---:|" * len(SNRS)]
    for fpr in (0.01, 0.001):
        L += ["", f"## {2 if fpr == 0.01 else 3}. threshold를 검증셋 정상의 {fpr * 100:g} %가 넘게 정했을 때", "",
              "**테스트셋 정상 스트림 20,000장에서 잰 FPR (%)와 시간당 경보 수 (회)**", "",
              "| 방법 | FPR (%) | 시간당 경보 수, 최근 10장 중 1장 이상 | 시간당 경보 수, 최근 10장 중 2장 이상 |", "|---|---:|---:|---:|"]
        for m in ("A", "B"):
            r = R[m][fpr]; L.append(f"| {NAME[m]} | {pct(r['test_fpr'])} | {r['alarms_h'][1]:.1f} | {r['alarms_h'][2]:.1f} |")
        for g in G:
            a, lo, hi = agg(R, g, lambda r: r[fpr]["test_fpr"]); b = agg(R, g, lambda r: r[fpr]["alarms_h"][1]); c = agg(R, g, lambda r: r[fpr]["alarms_h"][2])
            L.append(f"| {NAME[g]} | {pct(a)} ({pct(lo)}–{pct(hi)}) | {b[0]:.1f} ({b[1]:.1f}–{b[2]:.1f}) | {c[0]:.1f} ({c[1]:.1f}–{c[2]:.1f}) |")
        for kind in (1, 2):
            L += ["", f"**이상 종류 {kind} ({C.ANOM[kind]['name']}): 스냅샷 recall (%)** — 이상 신호가 1개 이상 보이는 스냅샷만", ""] + hdr
            L += rows(lambda r, s: r[fpr]["cells"][f"{kind}_{s}"]["recall"])
            L.append("| 보이는 스냅샷 수 (장) | " + " | ".join(str(R['A'][fpr]['cells'][f'{kind}_{s}']['n_vis']) for s in SNRS) + " |")
            L += ["", f"**이상 종류 {kind}: 이벤트 recall (%), 이벤트 40개, 10장 중 2장 이상**", ""] + hdr
            L += rows(lambda r, s: r[fpr]["cells"][f"{kind}_{s}"]["event_det"][2])
            L += ["", f"**이상 종류 {kind}: 정밀도 (%), 이상 스냅샷 비율 0.1 % 가정**", ""] + hdr
            L += rows(lambda r, s: r[fpr]["cells"][f"{kind}_{s}"]["precision"][0.001], rng=False)
    L += ["", "## 4. AUROC (threshold 없이 정상 대 이상 순서만 본 값)", ""]
    for kind in (1, 2):
        L += [f"**이상 종류 {kind}**", ""] + hdr + rows(lambda r, s: r["auroc"][f"{kind}_{s}"], fmt=lambda x: f"{x:.3f}", rng=False) + [""]
    vis = {k: float((ev["n_vis"][(ev["anom"] == k)] == 0).mean()) for k in (1, 2)}
    L += ["## 5. 기기가 켜져 있어도 아무것도 안 보이는 스냅샷 비율", "",
          f"이상 종류 1: {pct(vis[1])} %, 이상 종류 2: {pct(vis[2])} % (수신 폭 20 MHz 가 블루투스 채널의 약 1/4 만 보기 때문)", ""]
    (C.RUNS / "results.md").write_text("\n".join(L), encoding="utf-8")


# ---------------- 그림 ----------------
def show(ax, db, title, vmin=-3, vmax=20):
    im = ax.imshow(db, origin="lower", aspect="auto", extent=EXT, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.set_xlabel("시간 (ms)", fontsize=8); ax.set_ylabel("주파수 (MHz)", fontsize=8); ax.tick_params(labelsize=7)
    return im


def boxes(ax, bl, kinds, color, ls="--", lw=1.0):
    for b in bl:
        if b[0] in kinds:
            ax.add_patch(Rectangle((b[1] * 1e3, (C.FC + b[3]) / 1e6), (b[2] - b[1]) * 1e3, (b[4] - b[3]) / 1e6, fill=False, ec=color, ls=ls, lw=lw))


def fig1(ev):
    """데이터 예시: 정상 2장, 학습용 사각형 덩어리 2장, 테스트 이상 종류 1·2 각 2장(SNR 12, 3 dB)."""
    X = {n: np.load(C.DATA / f"{n}.npy", mmap_mode="r") for n in ("train_normal", "train_blob", "test_event")}
    B = {n: json.load(open(C.DATA / f"{n}_boxes.json")) for n in X}
    tn = read_csv("train_normal")
    i_w = int(np.where((tn["n_wifi"] >= 2) & (tn["n_bt"] >= 2))[0][0]); i_b = int(np.where((tn["n_wifi"] == 0) & (tn["n_bt"] >= 3))[0][0])
    pick = lambda a, s: int(np.where((ev["anom"] == a) & (ev["snr"] == s) & (ev["n_vis"] >= 2))[0][0])
    P = [("train_normal", i_w, "정상 (와이파이 2개 이상)"), ("train_normal", i_b, "정상 (블루투스 짧은 패킷만)"),
         ("train_blob", 0, "학습용 이상: 사각형 덩어리"), ("train_blob", 3, "학습용 이상: 사각형 덩어리"),
         ("test_event", pick(1, 12), "테스트 이상 1, SNR 12 dB"), ("test_event", pick(1, 3), "테스트 이상 1, SNR 3 dB"),
         ("test_event", pick(2, 12), "테스트 이상 2, SNR 12 dB"), ("test_event", pick(2, 3), "테스트 이상 2, SNR 3 dB")]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    for ax, (n, i, t) in zip(axs.ravel(), P):
        im = show(ax, C.to_db(X[n][i]), t)
        boxes(ax, B[n][i], ("wifi", "bt"), "cyan", ":", 0.8); boxes(ax, B[n][i], ("blob", "anom1", "anom2"), "lime", "--", 1.2)
    fig.colorbar(im, ax=axs, shrink=0.6, label="잡음 바닥 대비 세기 (dB)")
    fig.savefig(C.FIG / "fig1_examples.png", dpi=130); plt.close(fig)


def fig2(ev, ps):
    """픽셀차 과정: 테스트 이상 1(SNR 9 dB) 한 장."""
    st = np.load(C.RUNS / "pixel_stats.npz"); mu, sd = st["mu"], st["sd"]
    X = np.load(C.DATA / "test_event.npy", mmap_mode="r"); Bx = json.load(open(C.DATA / "test_event_boxes.json"))
    inside = lambda bl: all(-8e6 < b[3] and b[4] < 8e6 and 2e-3 < b[1] and b[2] < 18e-3 for b in bl if b[0] == "anom1")
    cand = np.where((ev["anom"] == 1) & (ev["snr"] == 9) & (ev["n_vis"] >= 2) & (ev["pos"] >= 2))[0]
    i = int(next(j for j in cand if inside(Bx[j])))                        # 이상 패킷이 그림 가장자리에 걸리지 않은 첫 예
    from pixeldiff import smooth
    x, xp = C.to_db(X[i]), C.to_db(X[i - 1]); z = (x - mu) / sd; d = x - xp
    zs = smooth(z, ps["B"]["rows"], ps["B"]["cols"]); ds = smooth(d, ps["A"]["rows"], ps["A"]["cols"])     # 상자가 밖으로 나가는 가장자리는 NaN(흰색)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    im0 = show(axs[0, 0], x, "1. 지금 스냅샷"); boxes(axs[0, 0], Bx[i], ("anom1",), "lime", "--", 1.2)
    show(axs[0, 1], mu, "2. 픽셀별 정상 평균 (학습셋 정상 10,000장)")
    im2 = axs[0, 2].imshow(sd, origin="lower", aspect="auto", extent=EXT, cmap="viridis"); axs[0, 2].set_title("3. 픽셀별 정상 표준편차 (dB)", fontsize=9); fig.colorbar(im2, ax=axs[0, 2])
    im3 = axs[0, 3].imshow(z, origin="lower", aspect="auto", extent=EXT, cmap="RdBu_r", vmin=-8, vmax=8); axs[0, 3].set_title("4. (나) 정상 평균과의 차 ÷ 표준편차", fontsize=9); fig.colorbar(im3, ax=axs[0, 3])
    im4 = axs[1, 0].imshow(zs, origin="lower", aspect="auto", extent=EXT, cmap="RdBu_r", vmin=-6, vmax=6)
    r, c = np.unravel_index(np.nanargmax(zs), zs.shape); tt = (c * C.HOP + C.NFFT / 2) / C.FS * 1e3; ff = (C.FC - C.FS / 2 + (r + 0.5) * C.DF_ROW) / 1e6
    axs[1, 0].plot(tt, ff, "k+", ms=14, mew=2); axs[1, 0].set_title(f"5. (나) 평활 {ps['B']['rows']}줄 × {ps['B']['cols']}칸, + = 최댓값 {np.nanmax(zs):.1f}", fontsize=9); fig.colorbar(im4, ax=axs[1, 0])
    show(axs[1, 1], xp, "6. 1초 전 스냅샷")
    im6 = axs[1, 2].imshow(d, origin="lower", aspect="auto", extent=EXT, cmap="RdBu_r", vmin=-12, vmax=12); axs[1, 2].set_title("7. (가) 지금 - 1초 전 (dB)", fontsize=9); fig.colorbar(im6, ax=axs[1, 2])
    im7 = axs[1, 3].imshow(ds, origin="lower", aspect="auto", extent=EXT, cmap="RdBu_r", vmin=-8, vmax=8)
    r, c = np.unravel_index(np.nanargmax(ds), ds.shape); tt = (c * C.HOP + C.NFFT / 2) / C.FS * 1e3; ff = (C.FC - C.FS / 2 + (r + 0.5) * C.DF_ROW) / 1e6
    axs[1, 3].plot(tt, ff, "k+", ms=14, mew=2); axs[1, 3].set_title(f"8. (가) 평활 {ps['A']['rows']}줄 × {ps['A']['cols']}칸, + = 최댓값 {np.nanmax(ds):.1f}", fontsize=9); fig.colorbar(im7, ax=axs[1, 3])
    for ax in axs.ravel(): ax.set_xlabel("시간 (ms)", fontsize=8); ax.set_ylabel("주파수 (MHz)", fontsize=8); ax.tick_params(labelsize=7)
    fig.colorbar(im0, ax=axs[0, :2], shrink=0.8, label="잡음 바닥 대비 세기 (dB)")
    fig.savefig(C.FIG / "fig2_pixeldiff_steps.png", dpi=130); plt.close(fig)


COL = {"A": "#1f77b4", "B": "#ff7f0e", "CNN": "#2ca02c", "CAE": "#9467bd", "ENS": "#d62728"}


def fig3(R):
    """SNR 별 스냅샷 recall. 실선 = threshold 를 검증셋 정상의 1 %가 넘게 정했을 때, 점선 = 0.1 %."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    for ax, kind in zip(axs, (1, 2)):
        ax.axvspan(2, 10, color="0.9", zorder=0); ax.text(6, 1.02, "정상 와이파이·블루투스의 SNR 범위 (2–10 dB)", ha="center", fontsize=8, color="0.35")
        for fpr, ls in ((0.01, "-"), (0.001, "--")):
            for m in ("A", "B"):
                ax.plot(SNRS, [R[m][fpr]["cells"][f"{kind}_{s}"]["recall"] for s in SNRS], ls, marker="o", color=COL[m], label=NAME[m] if fpr == 0.01 else None)
            for g in groups(R):
                pre = LEARNED[g][0]
                v = np.array([[R[m][fpr]["cells"][f"{kind}_{s}"]["recall"] for s in SNRS] for m in R if m[0] == pre and m[1:].isdigit()])
                ax.plot(SNRS, v.mean(0), ls, marker={"CNN": "s", "CAE": "^", "ENS": "D"}[g], color=COL[g], label=f"{NAME[g]} (시드 3개 평균)" if fpr == 0.01 else None)
                ax.fill_between(SNRS, v.min(0), v.max(0), color=COL[g], alpha=0.15)
        ax.set_title(f"테스트 이상 {kind}: {C.ANOM[kind]['name']}", fontsize=10); ax.set_xlabel("이상 신호 SNR (dB)"); ax.set_ylabel("스냅샷 recall"); ax.set_ylim(-0.02, 1.08); ax.set_xticks(SNRS); ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    h, l = axs[0].get_legend_handles_labels()
    h += [Line2D([], [], color="0.3", ls="-"), Line2D([], [], color="0.3", ls="--")]
    l += ["실선: threshold를 검증셋 정상의 1 %가 넘게 정했을 때", "점선: threshold를 검증셋 정상의 0.1 %가 넘게 정했을 때"]
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(C.FIG / "fig3_recall_vs_snr.png", dpi=130); plt.close(fig)


def fig4(M):
    """1시간 시연 스트림의 이상 점수: (나), CNN 시드 0, CAE 시드 0, 앙상블 시드 0. 위 3줄은 1시간 전체, 아래 3줄은 이벤트 앞뒤 확대."""
    from make_data import DEMO_EVENTS
    dm = read_csv("test_demo"); t = np.arange(len(dm["idx"]))
    rows = [(m, nm) for m, nm in (("B", "B"), ("C0", "CNN"), ("E0", "CAE"), ("N0", "ENS")) if m in M]
    fig = plt.figure(figsize=(15, 4.4 * len(rows)), constrained_layout=True); gs = fig.add_gridspec(2 * len(rows), 3)
    for ri, (m, nm) in enumerate(rows):
        ax = fig.add_subplot(gs[ri, :]); s = M[m]["s"]["test_demo"]
        ax.plot(t, s, lw=0.5, color="0.3")
        for k, (t0, a, snr) in enumerate(DEMO_EVENTS):
            ax.axvspan(t0 - 0.5, t0 + 9.5, color="red", alpha=0.25)
            ax.text(t0 + 25, 0.97, f"이상 {a}, SNR {snr:g} dB", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color="red")
        ax.axhline(M[m]["thr"][0.01], color="#ff7f0e", lw=1, label="threshold: 검증셋 정상의 1 %가 넘는 값"); ax.axhline(M[m]["thr"][0.001], color="#d62728", lw=1, ls="--", label="threshold: 검증셋 정상의 0.1 %가 넘는 값")
        ax.set_title(f"{NAME[nm]}{' 시드 0' if m[1:].isdigit() else ''}: 1시간 시연 스트림 (1초에 한 장, 3,600장)", fontsize=10); ax.set_xlabel("시간 (초)"); ax.set_ylabel("이상 점수"); ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(0, len(t))
        for k, (t0, a, snr) in enumerate(DEMO_EVENTS):
            az = fig.add_subplot(gs[len(rows) + ri, k]); sl = slice(t0 - 20, t0 + 30)
            az.plot(t[sl], s[sl], "-o", ms=3, lw=0.8, color="0.3"); az.axvspan(t0 - 0.5, t0 + 9.5, color="red", alpha=0.2)
            nv = dm["n_vis"][sl]; z = (nv == 0) & (dm["anom"][sl] > 0)
            az.plot(t[sl][z], s[sl][z], "x", color="blue", ms=8, mew=2, label="기기 켜짐, 보이는 패킷 0개")
            az.axhline(M[m]["thr"][0.01], color="#ff7f0e", lw=1); az.axhline(M[m]["thr"][0.001], color="#d62728", lw=1, ls="--")
            az.set_title(f"{NAME[nm][:3]} 확대: 이상 {a}, SNR {snr:g} dB", fontsize=9); az.set_xlabel("시간 (초)", fontsize=8); az.tick_params(labelsize=7)
            if k == 0: az.legend(fontsize=7, loc="upper left")
    fig.savefig(C.FIG / "fig4_demo_stream.png", dpi=130); plt.close(fig)


def fig5(M):
    """테스트셋 정상 스트림에서 점수가 가장 높았던 정상 스냅샷(오경보) 3장씩: (나), CNN 시드 0, CAE 시드 0, 앙상블 시드 0."""
    X = np.load(C.DATA / "test_normal.npy", mmap_mode="r"); Bx = json.load(open(C.DATA / "test_normal_boxes.json"))
    P = np.load(C.RUNS / "scores_pixel.npz")
    rows = [(m, nm) for m, nm in (("B", "B"), ("C0", "CNN"), ("E0", "CAE"), ("N0", "ENS")) if m in M]
    fig, axs = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)), constrained_layout=True)
    for ri, (m, nm) in enumerate(rows):
        s = M[m]["s"]["test_normal"]; top = np.argsort(-np.nan_to_num(s, nan=-1e9))[:3]
        for ax, i in zip(axs[ri], top):
            im = show(ax, C.to_db(X[i]), f"{NAME[nm][:3]} 오경보: 점수 {s[i]:.2f}\nthreshold {M[m]['thr'][0.01]:.2f}(검증셋 정상의 1 %가 넘는 값)")
            boxes(ax, Bx[i], ("wifi",), "cyan", ":", 1.0); boxes(ax, Bx[i], ("bt",), "white", ":", 1.0)
            if m == "B":
                r, c = map(int, P["B/test_normal/loc"][i]); ax.plot((c * C.HOP + C.NFFT / 2) / C.FS * 1e3, (C.FC - C.FS / 2 + (r + 0.5) * C.DF_ROW) / 1e6, "g+", ms=14, mew=2)
    fig.colorbar(im, ax=axs, shrink=0.6, label="잡음 바닥 대비 세기 (dB)")
    fig.savefig(C.FIG / "fig5_false_alarms.png", dpi=130); plt.close(fig)


def fig6(ev, sel):
    """CAE(시드 0) 복원 예시: 정상(와이파이 있음), 테스트 이상 1 SNR 6 dB, 테스트 이상 2 SNR 12 dB.
    칸: 입력(표준점수 지도), 복원, 입력 - 복원의 양수 부분, 그것을 평활한 지도(+ = 최댓값 = 이상 점수)."""
    if "CAE" not in sel: return
    import torch
    from cae import CAE
    from cnn import Prep
    from pixeldiff import smooth
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wf = C.RUNS / "cae_s0.pt" if (C.RUNS / "cae_s0.pt").exists() else C.MODELS / "cae_s0.pt"     # 학습을 건너뛰었으면 저장소의 학습된 모델
    net = CAE(); net.load_state_dict(torch.load(wf, map_location=dev)); net.to(dev).eval()
    st = np.load(C.RUNS / "pixel_stats.npz"); mu, sd = torch.tensor(st["mu"], device=dev), torch.tensor(st["sd"], device=dev)
    r_, c_ = sel["CAE"]["0"]["rows"], sel["CAE"]["0"]["cols"]
    Xn = np.load(C.DATA / "test_normal.npy", mmap_mode="r"); Bn = json.load(open(C.DATA / "test_normal_boxes.json")); tn = read_csv("test_normal")
    Xe = np.load(C.DATA / "test_event.npy", mmap_mode="r"); Be = json.load(open(C.DATA / "test_event_boxes.json"))
    inside = lambda bl, k: all(-8e6 < b[3] and b[4] < 8e6 and 2e-3 < b[1] and b[2] < 18e-3 for b in bl if b[0] == f"anom{k}")
    i_n = int(np.where((tn["n_wifi"] >= 1) & (tn["n_bt"] >= 2))[0][0])
    pick = lambda a, s: int(next(j for j in np.where((ev["anom"] == a) & (ev["snr"] == s) & (ev["n_vis"] >= 1))[0] if inside(Be[j], a)))
    P = [(Xn[i_n], Bn[i_n], "정상 (와이파이·블루투스)"), (Xe[pick(1, 6)], Be[pick(1, 6)], "테스트 이상 1, SNR 6 dB"), (Xe[pick(2, 12)], Be[pick(2, 12)], "테스트 이상 2, SNR 12 dB")]
    fig, axs = plt.subplots(3, 4, figsize=(16, 10.5), constrained_layout=True)
    with torch.no_grad():
        for ri, (u8, bl, title) in enumerate(P):
            db = torch.from_numpy(np.array(u8)).to(dev).float() / 255 * (C.DB_HI - C.DB_LO) + C.DB_LO
            x = (((db - mu) / sd).clamp(-4, 12) / 4)[None, None]; y = net(x)
            xi, yi = x[0, 0].cpu().numpy(), y[0, 0].cpu().numpy(); res = np.clip(xi - yi, 0, None); sm = smooth(res, r_, c_)
            for j, (img, t2, cm, lim) in enumerate([(xi, f"{title}: 1. 입력 (표준점수 ÷ 4)", "RdBu_r", 2.5), (yi, "2. CAE 복원", "RdBu_r", 2.5),
                                                    (res, "3. 입력 - 복원 (양수 부분)", "magma", 2.0), (sm, f"4. 3을 평활 {r_}줄 × {c_}칸, + = 최댓값 {np.nanmax(sm):.3f}", "magma", None)]):
                ax = axs[ri, j]; im = ax.imshow(img, origin="lower", aspect="auto", extent=EXT, cmap=cm, interpolation="nearest",
                                                vmin=(-lim if cm == "RdBu_r" else 0) if lim else None, vmax=lim)
                ax.set_title(t2, fontsize=9); ax.set_xlabel("시간 (ms)", fontsize=8); ax.set_ylabel("주파수 (MHz)", fontsize=8); ax.tick_params(labelsize=7); fig.colorbar(im, ax=ax)
                if j == 0: boxes(ax, bl, ("anom1", "anom2"), "lime", "--", 1.2)
                if j == 3:
                    rr, cc = np.unravel_index(np.nanargmax(sm), sm.shape)
                    ax.plot((cc * C.HOP + C.NFFT / 2) / C.FS * 1e3, (C.FC - C.FS / 2 + (rr + 0.5) * C.DF_ROW) / 1e6, "c+", ms=14, mew=2)
    fig.savefig(C.FIG / "fig6_cae_examples.png", dpi=130); plt.close(fig)


def main():
    C.FIG.mkdir(parents=True, exist_ok=True)
    M, ps, sel = load_scores(); ev = read_csv("test_event")
    R = evaluate(M, ev)
    json.dump({m: {str(k): v for k, v in r.items()} for m, r in R.items()}, open(C.RUNS / "results.json", "w"), ensure_ascii=False, indent=1)
    write_md(R, ps, sel, ev)
    fig1(ev); fig2(ev, ps); fig3(R); fig4(M); fig5(M); fig6(ev, sel)
    print("EVALUATE_DONE", flush=True)


if __name__ == "__main__":
    main()
