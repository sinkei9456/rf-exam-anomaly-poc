"""두 번째 실험 문서(doc/00_overview_ver2.md, 교정본 00_overview_rev2.md)의 설명 그림 두 장 (2026-11-19). 학습·평가 결과는 바꾸지 않는다.

  EXAM_SITE=park   python src/site_figures.py snr     → doc/fig/site/fig_snr_example.png
      같은 공원 배경 20 ms(테스트셋 파일에서 실제 신호가 가장 적은 구간)에 같은 이상 신호 1(블루투스 이어폰 오디오 전송) 패킷을 SNR 0·6·12 dB 로 넣은 스펙트로그램 3장과,
      패킷이 지나가는 시간 칸에서 주파수별 밝기를 평균한 선 그림 1장. 정상 와이파이·블루투스는 넣지 않는다.
  EXAM_SITE=indoor python src/site_figures.py dense   → doc/fig/site/fig_indoor_dense.png
      실내 테스트셋 파일 05 의 190–250 ms(합성 없음) 스펙트로그램 3장과, 정상 스냅샷 중 판정 임계값을 넘은 것의 시작 위치 분포.
"""
import sys, json
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C
import evaluate as E
import matplotlib.pyplot as plt

LABEL = "배경 잡음 세기 대비 밝기 (dB)"


def fig_snr():
    assert C.SITE == "park"
    a, sd = C.background("test")
    # 실제 신호가 가장 적은 20 ms: 10 ms 간격으로 훑어 가장 밝은 픽셀이 가장 어두운 위치
    offs = list(range(0, len(a[0]) - C.N + 1, C.N // 2))
    off = min(offs, key=lambda o: float(C.spec_db(np.asarray(a[0][o: o + C.N]) / sd).max()))
    x0 = (np.asarray(a[0][off: off + C.N]) / sd).astype(np.complex64)
    # 패킷 하나가 통째로(2.87 ms) 그림 가운데(가장자리에서 6 MHz 안쪽)에 보이는 패킷 배치를 고른다. SNR 은 크기만 바꾸므로 세 SNR 의 위치가 같다.
    def full(bl): return [q for q in bl if q[2] - q[1] > 2.8e-3 and abs((q[3] + q[4]) / 2) < 4e6]
    seed = next(k for k in range(1000) if full(C.add_anomaly(x0.copy(), np.random.default_rng(k), 1, 12.0)[0]))
    print(f"배경 위치 {off / C.FS * 1e3:.0f} ms, 패킷 배치 난수 {seed}")
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    prof = {}
    for ax, snr in zip(axs[:3], (0, 6, 12)):
        x = x0.copy(); bl, _, _ = C.add_anomaly(x, np.random.default_rng(seed), 1, float(snr))
        db = C.spec_db(x); im = E.show(ax, db, f"({'abc'[(0, 6, 12).index(snr)]}) SNR {snr} dB")
        E.boxes(ax, bl, ("anom1",), "lime", "--", 1.0)
        b = full(bl)[0]                                                   # 통째로 보이는 패킷
        c0 = int(np.ceil((b[1] * C.FS - C.NFFT / 2) / C.HOP)) + 1; c1 = int(np.floor((b[2] * C.FS - C.NFFT / 2) / C.HOP)) - 1
        prof[snr] = (10 * np.log10((10 ** (db[:, c0:c1 + 1] / 10)).mean(1)), b)
    fig.colorbar(im, ax=axs[:3], shrink=0.85, label=LABEL)
    f = (C.FC - C.FS / 2 + (np.arange(C.NF) + 0.5) * C.DF_ROW) / 1e6
    for snr, col in zip((0, 6, 12), ("#1f77b4", "#ff7f0e", "#d62728")):
        p, b = prof[snr]; axs[3].plot(f, p, color=col, lw=1, label=f"SNR {snr} dB")
    b = prof[12][1]; lo, hi = (C.FC + b[3]) / 1e6, (C.FC + b[4]) / 1e6
    axs[3].axvspan(lo, hi, color="0.85", zorder=0)
    for v in (0, 3, 7, 12.3): axs[3].axhline(v, color="0.5", ls=":", lw=0.8)
    axs[3].set_xlim(lo - 3, hi + 3); axs[3].set_ylim(-4, 16)
    axs[3].text(hi + 0.2, 13.0, "점선: 0, 3, 7, 12.3 dB", fontsize=7, color="0.4")
    axs[3].set_title("(d) 패킷이 지나가는 시간 칸의 주파수별 평균 밝기", fontsize=9); axs[3].set_xlabel("주파수 (MHz)", fontsize=8); axs[3].set_ylabel(LABEL, fontsize=8)
    axs[3].legend(fontsize=8, loc="upper left"); axs[3].tick_params(labelsize=7)
    fig.savefig(C.FIG / "fig_snr_example.png", dpi=120); plt.close(fig)
    for snr in (0, 6, 12):
        p, b = prof[snr]; r = (f >= (C.FC + b[3]) / 1e6 + 0.1) & (f <= (C.FC + b[4]) / 1e6 - 0.1); o = (np.abs(f - (C.FC + (b[3] + b[4]) / 2) / 1e6) > 2) & (np.abs(f - (C.FC + (b[3] + b[4]) / 2) / 1e6) < 4)
        print(f"SNR {snr} dB: 패킷 안 평균 {10 * np.log10(np.mean(10 ** (p[r] / 10))):.2f} dB, 패킷 중심에서 2–4 MHz 떨어진 곳 평균 {10 * np.log10(np.mean(10 ** (p[o] / 10))):.2f} dB")


def fig_dense():
    assert C.SITE == "indoor"
    a, sd = C.background("test")
    tn = E.read_csv("test_normal"); off = tn["bg_offset"].astype(np.int64) % 10 ** 8
    Q = np.load(C.RUNS / "scores_cae65k.npz"); sel = json.load(open(C.RUNS / "cae65k_select.json"))
    P = np.load(C.RUNS / "scores_pixel.npz"); ps = json.load(open(C.RUNS / "pixel_select.json"))
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    for ax, t0, k in zip(axs[:3], (190, 210, 230), "abc"):
        s = int(t0 * 1e-3 * C.FS); im = E.show(ax, C.spec_db(np.asarray(a[0][s: s + C.N]) / sd), f"({k}) 테스트셋 파일 05, {t0}–{t0 + 20} ms")
        if k == "b":                                                   # 오탐 점수의 최댓값이 몰린 곳: 2,451 MHz 에서 211·214·217 ms 에 되풀이된 실제 신호
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((0.6, 2450.4), 8.0, 1.2, fill=False, ec="white", ls="--", lw=1.2))
    fig.colorbar(im, ax=axs[:3], shrink=0.85, label=LABEL)
    bins = np.arange(0, 1001, 50)
    for name, s, t, col in (("CAE, 파라미터 65,265개, 시드 0", Q["E0/test_normal"], sel["0"]["thr_fpr01"], "#6a3d9a"),
                            ("(나) 정상 기준 차분", P["B/test_normal"], ps["B"]["thr_fpr01"], "#ff7f0e")):
        fl = s > t; h = np.histogram(off[fl] / C.FS * 1e3, bins=bins)[0]
        axs[3].bar(bins[:-1] + (0 if col == "#6a3d9a" else 22), h, width=22, align="edge", color=col, label=f"{name} ({int(fl.sum())}장)")
        print(name, "임계값을 넘은 정상 스냅샷", int(fl.sum()), "시작 위치 200–250 ms", int(h[4]))
    axs[3].set_title("(d) 임계값을 넘은 정상 스냅샷의 시작 위치 (50 ms 구간별)", fontsize=9)
    axs[3].set_xlabel("테스트셋 파일 05 안에서 20 ms를 잘라 낸 시작 위치 (ms)", fontsize=8); axs[3].set_ylabel("스냅샷 수 (장)", fontsize=8)
    axs[3].legend(fontsize=8); axs[3].tick_params(labelsize=7)
    fig.savefig(C.FIG / "fig_indoor_dense.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    {"snr": fig_snr, "dense": fig_dense}[sys.argv[1]]()
