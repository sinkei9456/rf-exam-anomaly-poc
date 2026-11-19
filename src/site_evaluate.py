"""장소별 실험의 테스트셋 평가와 부록 그림 (2026-11-19). 사용: EXAM_SITE=<장소> python src/site_evaluate.py

threshold, 점수 상자 크기, CAE 에포크는 모두 그 장소 검증셋에서 정한 값만 쓴다(pixel_select.json, cae<크기>_select.json).
테스트셋은 결과를 잴 때만 쓴다. 지표 계산은 evaluate.py 의 evaluate() 를 그대로 쓴다.
  - 스냅샷 FPR: 테스트셋 정상 스트림 20,000장 중 threshold 를 넘은 비율
  - 스냅샷 recall: 이상 신호가 1개 이상 보이는 테스트 이상 스냅샷 중 threshold 를 넘은 비율(종류 2 × SNR 5단계)
  - 이벤트 recall: 기기가 10초 켜진 이벤트(1초에 한 장, 10장) 중 k장 이상 threshold 를 넘은 비율
  - 시간당 경보 수: 정상 스트림에서 '최근 10장 중 k장 이상'이 처음 참이 되는 횟수 / 5.56시간
가벼운 정도: 학습 파라미터 수, 모델 파일 크기, GPU 학습 시간(시드 평균), CPU 코어 1개에서 스냅샷 1장의 점수를 내는 시간(중앙값).
출력: runs/site/<장소>/results.json, runs/site/<장소>/light.json,
      doc/fig/site/<장소>_a_background.png(배경만 9장), _b_normal.png(정상 9장), _c_anom1.png(이상 1, 9장), _d_anom2.png(이상 2, 9장)
"""
import sys, json, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C
import evaluate as E
import matplotlib.pyplot as plt

assert C.SITE, "EXAM_SITE 를 정해야 한다(indoor, indoor_droneoff, park)"
SIZES = ["65k", "21k", "6k"]
ROLE_KO = {"train": "학습셋", "val": "검증셋", "test": "테스트셋"}


def load_models():
    P = np.load(C.RUNS / "scores_pixel.npz"); ps = json.load(open(C.RUNS / "pixel_select.json"))
    M = {m: dict(s={n: P[f"{m}/{n}"] for n in ("test_normal", "test_event")}, thr={0.01: ps[m]["thr_fpr1"], 0.001: ps[m]["thr_fpr01"]}) for m in ("A", "B")}
    sel = {}
    for z in SIZES:
        Q = np.load(C.RUNS / f"scores_cae{z}.npz"); sel[z] = json.load(open(C.RUNS / f"cae{z}_select.json"))
        for s in sorted(sel[z]):
            M[f"cae{z}_s{s}"] = dict(s={n: Q[f"E{s}/{n}"] for n in ("test_normal", "test_event")},
                                    thr={0.01: sel[z][s]["thr_fpr1"], 0.001: sel[z][s]["thr_fpr01"]})
    return M, ps, sel


def light(ps, sel):
    """가벼운 정도. CPU 시간은 코어 1개(torch.set_num_threads(1)), 배치 1, 테스트셋 정상 스냅샷 200장의 중앙값."""
    import torch
    from scipy import ndimage
    from cae import CAE, SIZES as CH
    torch.set_num_threads(1)
    X = np.load(C.DATA / "test_normal.npy", mmap_mode="r")[:200]
    st = np.load(C.RUNS / "pixel_stats.npz"); mu, sd = st["mu"], st["sd"]
    out = {}
    for m in ("A", "B"):
        r, c = ps[m]["rows"], ps[m]["cols"]; ts = []
        for i in range(1, len(X)):
            t = time.perf_counter(); x = C.to_db(X[i])
            d = x - C.to_db(X[i - 1]) if m == "A" else (x - mu) / sd
            ndimage.uniform_filter(d, size=(r, c), mode="nearest").max(); ts.append(time.perf_counter() - t)
        out[m] = dict(n_params=0, cpu_ms=float(np.median(ts) * 1e3))
    tmu, tsd = torch.tensor(mu), torch.tensor(sd)
    for z in SIZES:
        net = CAE(tuple(CH[z])); net.load_state_dict(torch.load(C.RUNS / f"cae{z}_s0.pt", map_location="cpu")); net.eval()
        r, c = sel[z]["0"]["rows"], sel[z]["0"]["cols"]; ts = []
        with torch.no_grad():
            for i in range(len(X)):
                t = time.perf_counter()
                db = torch.from_numpy(np.array(X[i])).float() / 255 * (C.DB_HI - C.DB_LO) + C.DB_LO
                x = (((db - tmu) / tsd).clamp(-4, 12) / 4)[None, None]
                torch.nn.functional.avg_pool2d((x - net(x)).clamp(min=0), (r, c), stride=1).max().item(); ts.append(time.perf_counter() - t)
        v = sel[z]
        out[f"cae{z}"] = dict(n_params=int(v["0"]["n_params"]), channels=v["0"]["channels"], file_bytes=int((C.RUNS / f"cae{z}_s0.pt").stat().st_size),
                              train_seconds=float(np.mean([v[s]["train_seconds"] for s in v])), epochs=len(v["0"]["history"]),
                              cpu_ms=float(np.median(ts[10:]) * 1e3))
    return out


# ---------------- 부록 그림 ----------------
def grid(P, path, suptitle):
    """P = [(dB 그림, 제목, 넣은 신호 박스 목록 또는 None)] 9개 → 3 × 3 그림."""
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 10), constrained_layout=True)
    for ax, (db, title, bl) in zip(axs.ravel(), P):
        im = E.show(ax, db, title)
        if bl:
            E.boxes(ax, bl, ("wifi", "bt"), "cyan", ":", 0.8); E.boxes(ax, bl, ("anom1", "anom2"), "lime", "--", 1.2)
    fig.colorbar(im, ax=axs, shrink=0.6, label="배경 잡음 세기 대비 밝기 (dB)")
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(path, dpi=110); plt.close(fig)


def fig_background():
    """배경만(합성 신호 없음) 9장: 장소의 모든 파일에 걸쳐 시간 간격을 고르게 둔 8장 + 가장 센 신호가 있는 1장."""
    C.background("train"); sd = C.NOISE_STATE["sd"]
    files = [(r, k, f) for r in ("train", "val", "test") for k, f in enumerate(C.BG_FILES[r])]
    arrs = [np.load(f, mmap_mode="r") for (_, _, f) in files]
    slots = [(j, o) for j, a in enumerate(arrs) for o in range(0, len(a) - C.N + 1, C.N)]          # 겹치지 않는 20 ms 위치
    pw = np.array([float(np.mean(np.abs(np.asarray(arrs[j][o:o + C.N])) ** 2)) for j, o in slots])
    strong = int(pw.argmax())
    even = [int(round(v)) for v in np.linspace(0, len(slots) - 1, 9)]
    pick = [q for q in even if q != strong][:8] + [strong]
    P = []
    for q in pick:
        j, o = slots[q]; r, k, f = files[j]
        db = C.spec_db(np.asarray(arrs[j][o:o + C.N]) / sd)
        tag = " (가장 센 신호)" if q == strong else ""
        P.append((db, f"{ROLE_KO[r]} 파일 {f.stem}, {o / C.FS * 1e3:.0f}–{(o + C.N) / C.FS * 1e3:.0f} ms{tag}", None))
    grid(P, C.FIG / f"{C.SITE}_a_background.png", f"{C.SITE_NAME}: 받은 IQ 데이터 그대로 (합성 신호 없음)")


def fig_normal():
    X = np.load(C.DATA / "test_normal.npy", mmap_mode="r"); B = json.load(open(C.DATA / "test_normal_boxes.json")); tn = E.read_csv("test_normal")
    idx = sorted(np.random.default_rng(0).choice(len(X), 9, replace=False))
    P = [(C.to_db(X[i]), f"테스트셋 정상 {i}번: 와이파이 {int(tn['n_wifi'][i])}개, 블루투스 {int(tn['n_bt'][i])}개", B[i]) for i in idx]
    grid(P, C.FIG / f"{C.SITE}_b_normal.png", f"{C.SITE_NAME}: 정상 스냅샷 (배경 + 약한 와이파이·블루투스 합성, 점선 = 넣은 정상 신호)")


def fig_anom(kind, ev):
    X = np.load(C.DATA / "test_event.npy", mmap_mode="r"); B = json.load(open(C.DATA / "test_event_boxes.json"))
    P = []
    for s in (12, 6, 0):
        q = np.where((ev["anom"] == kind) & (ev["snr"] == s) & (ev["n_vis"] >= 1))[0]
        evs = []
        for i in q:                                                    # 서로 다른 이벤트에서 처음 보이는 스냅샷 3장
            if ev["event"][i] not in [ev["event"][j] for j in evs]: evs.append(int(i))
            if len(evs) == 3: break
        P += [(C.to_db(X[i]), f"SNR {s} dB, 이벤트 {int(ev['event'][i])}, 보이는 패킷 {int(ev['n_vis'][i])}개", B[i]) for i in evs]
    grid(P, C.FIG / f"{C.SITE}_{'c' if kind == 1 else 'd'}_anom{kind}.png",
         f"{C.SITE_NAME}: 테스트 이상 {kind} ({C.ANOM[kind]['name']}), 초록 점선 = 넣은 이상 패킷, 하늘색 점선 = 넣은 정상 신호")


def main():
    C.FIG.mkdir(parents=True, exist_ok=True)
    M, ps, sel = load_models(); ev = E.read_csv("test_event")
    R = E.evaluate(M, ev)
    json.dump({m: {str(k): v for k, v in r.items()} for m, r in R.items()}, open(C.RUNS / "results.json", "w"), ensure_ascii=False, indent=1)
    L = light(ps, sel); json.dump(L, open(C.RUNS / "light.json", "w"), ensure_ascii=False, indent=1)
    for m, v in L.items(): print(m, v, flush=True)
    fig_background(); fig_normal(); fig_anom(1, ev); fig_anom(2, ev)
    print("SITE_EVALUATE_DONE", flush=True)


if __name__ == "__main__":
    main()
