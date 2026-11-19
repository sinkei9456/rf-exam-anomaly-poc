"""학습된 모델로 이상 탐지 추론 (2026-11-18, 장소 폴더 추론 2026-11-19). 학습 코드를 돌리지 않고 모델 파일만으로 쓴다. GPU 가 없으면 CPU 로 돈다.

두 가지 방식:
  --site DIR  (두 번째 실험 방식, 새 장소에 쓰는 방법) src/site_train.py 가 그 장소의 정상 IQ 데이터로 만든 장소 폴더를 읽는다.
              새 IQ 데이터를 그 폴더에 적힌 배경 잡음 세기로 나누고(학습 때와 같은 전처리) 그 장소의 CAE 나 (나) 픽셀 차이로 판정한다.
              --model cae(기본) | pixel, --cae 65k_s0 처럼 CAE 크기·시드를 고른다(기본은 site.json 의 default_cae).
              --noise-ref 를 주면 폴더의 배경 잡음 세기 대신 그 데이터로 다시 잰다.
  (--site 없음) 첫 번째 실험 방식. 아래 models/ 의 공원 배경 모델을 쓴다.

필요한 파일(models/):
  cnn_s0.pt, cnn_s1.pt, cnn_s2.pt   (다) CNN 가중치(시드 0·1·2)
  cae_s0.pt, cae_s1.pt, cae_s2.pt   (라) CAE(합성곱 오토인코더) 가중치(시드 0·1·2)
  pixel_stats.npz                   학습셋 정상 스냅샷 10,000장의 픽셀별 평균·표준편차(dB). CNN·CAE 입력과 (나) 픽셀차에 씀
  thresholds.json                   threshold 두 가지(검증셋 정상 스냅샷의 1 %·0.1 %가 넘는 값), 평활 크기, 학습 기록,
                                    (마) 앙상블의 눈금 맞춤 값(검증셋 정상 점수의 중앙값과 1 %가 넘는 값)

입력(셋 중 하나):
  --iq FILE         IQ 데이터. 20 MS/s 로 1초에 한 번 받은 20 ms 조각(400,000 표본)을 시간 순서대로 이어 붙인 파일.
                    .npy(complex), .cf32(complex64 raw), .cs8 또는 .sigmf-data(HackRF int8, I·Q 번갈아).
                    수신 조건은 학습과 같아야 한다: HackRF, 중심 2,449.5 MHz, 20 MS/s, 이득 고정.
  --snapshots FILE  이미 만든 스냅샷(uint8, 장수 × 256 × 97, src/make_data.py 출력과 같은 형식)
  --eval-sets       src/make_data.py 로 만든 검증셋·테스트셋의 점수를 저장소의 학습된 모델로 계산해 runs/ 에 저장한다(학습 없이 결과 재현).

잡음 세기 맞추기(--iq 일 때): 학습 때처럼 잡음 전력을 1 로 맞춘다.
  --noise-ref FILE  정상일 때 받은 IQ 데이터(권장: 시험 전 같은 장소·같은 수신 설정). 평균 전력으로 맞춘다(학습 때와 같은 계산).
  주지 않으면 입력에서 센 표본(평균 전력의 10배 넘는 것)을 뺀 평균 전력으로 맞추고 경고를 낸다.

판정:
  --model cnn|cae|ens|pixel  cnn = (다) CNN(기본), cae = (라) CAE, ens = (마) CNN·CAE 앙상블, pixel = (나) 정상 평균과의 픽셀차
                     앙상블은 같은 시드의 CNN·CAE 점수를 검증셋 정상으로 눈금 맞춤한 뒤 큰 값을 쓴다(src/ensemble.py).
  --seed 0|1|2       CNN·CAE·앙상블 시드(기본 0). 앙상블 시드 k = CNN 시드 k + CAE 시드 k
  --fpr 0.001|0.01   0.001 = 검증셋 정상 스냅샷의 0.1 %가 넘게 정한 threshold(기본), 0.01 = 1 %가 넘게 정한 threshold
  --k 2 --n 10       최근 n 장 중 k 장 이상이 이상이면 경보(기본 10장 중 2장)
  --interval 1.0     스냅샷 사이 간격(초). 출력의 시각 칸에만 쓴다.
출력: --out CSV(기본 runs/predict.csv). 칸: 번호, 시각(초), 이상 점수, threshold, 이상 판정(0/1), 경보(0/1).
화면에는 이상으로 판정한 스냅샷 수와 경보가 울린 시간 범위를 적는다.

예:
  python src/predict.py --site sites/indoor --iq new_capture.cs8
  python src/predict.py --site sites/indoor --iq new_capture.cs8 --model pixel
  python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy
  python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy --model cae
  python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy --model ens
  python src/predict.py --snapshots data/test_demo.npy
  python src/predict.py --eval-sets
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SETS = ["val_normal", "val_blob", "test_normal", "test_event", "test_demo"]


# ---------------- 모델 파일 ----------------
def load_meta():
    st = np.load(C.MODELS / "pixel_stats.npz")
    return st["mu"].astype(np.float32), st["sd"].astype(np.float32), json.load(open(C.MODELS / "thresholds.json", encoding="utf-8"))


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn(seed, dev):
    import torch
    from cnn import make_net
    net = make_net()
    net.load_state_dict(torch.load(C.MODELS / f"cnn_s{seed}.pt", map_location=dev))
    return net.to(dev).eval()


def load_cae(seed, dev):
    import torch
    from cae import CAE
    net = CAE()
    net.load_state_dict(torch.load(C.MODELS / f"cae_s{seed}.pt", map_location=dev))
    return net.to(dev).eval()


def _z(u8_batch, mu_t, sd_t, dev):
    """uint8 스냅샷 → 표준점수 지도(학습의 src/cnn.py Prep 과 같은 처리)."""
    import torch
    db = torch.from_numpy(np.array(u8_batch)).to(dev).float() / 255 * (C.DB_HI - C.DB_LO) + C.DB_LO
    return (((db - mu_t) / sd_t).clamp(-4, 12) / 4).unsqueeze(1)


def cnn_scores(net, u8, mu, sd, dev, bs=512):
    """(다) CNN 이상 점수(로짓)."""
    import torch
    mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(u8), bs):
            out.append(net(_z(u8[i:i + bs], mu_t, sd_t, dev)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def cae_scores(net, u8, mu, sd, rows, cols, dev, bs=256):
    """(라) CAE 이상 점수: 입력 - 복원의 양수 부분을 rows 줄 × cols 칸 상자로 평균한 뒤 최댓값(학습의 src/cae.py 와 같은 계산)."""
    import torch, torch.nn.functional as F
    mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(u8), bs):
            x = _z(u8[i:i + bs], mu_t, sd_t, dev); r = (x - net(x)).clamp(min=0)
            out.append(F.avg_pool2d(r, (rows, cols), stride=1).flatten(1).max(1).values.float().cpu().numpy())
    return np.concatenate(out)


def pixel_scores(u8, mu, sd, rows, cols):
    from pixeldiff import score_maps
    s, _ = score_maps((C.to_db(u8) - mu) / sd, rows, cols)
    return s


# ---------------- IQ 읽기 ----------------
class Int8IQ:
    """HackRF int8 IQ 파일(I·Q 번갈아)을 복소수 배열처럼 쓴다. 메모리에 다 올리지 않고 자른 부분만 complex64 로 바꾼다(÷ 128)."""
    def __init__(self, p):
        self.r = np.memmap(p, dtype=np.int8, mode="r"); self.n = len(self.r) // 2

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        a, b, st = s.indices(self.n)
        if st != 1: raise ValueError("Int8IQ 는 연속 구간만 자를 수 있음")
        r = np.asarray(self.r[2 * a: 2 * b], dtype=np.float32)
        return ((r[0::2] + 1j * r[1::2]) / 128).astype(np.complex64)


def open_iq(path):
    """IQ 파일을 연다. .npy(복소수)·.cf32(complex64)는 mmap, .cs8·.sigmf-data(HackRF int8)는 int8 ÷ 128.
    장소 폴더의 배경 잡음 세기도 이 방식으로 읽은 값의 단위다."""
    p = Path(path)
    if p.suffix == ".npy":
        x = np.load(p, mmap_mode="r")
        if not np.iscomplexobj(x): raise SystemExit(f"{p}: 복소수 배열이 아님")
        return x
    if p.suffix == ".cf32":
        return np.memmap(p, dtype=np.complex64, mode="r")
    if p.suffix in (".cs8", ".sigmf-data"):
        return Int8IQ(p)
    raise SystemExit(f"{p}: 읽을 수 없는 형식(.npy, .cf32, .cs8, .sigmf-data)")


read_iq = open_iq


def chunks(x):
    n = len(x) // C.N
    if n == 0: raise SystemExit(f"IQ 길이 {len(x)} 가 20 ms 조각 한 개({C.N} 표본)보다 짧음")
    if len(x) % C.N: print(f"경고: IQ 길이가 {C.N} 의 배수가 아니라 끝의 {len(x) % C.N} 표본을 버림", flush=True)
    return n


def noise_power_ref(x):
    """정상 IQ 데이터의 평균 전력(평균을 뺀 뒤). 학습 때 배경 노이즈를 맞춘 계산과 같다."""
    tot, cnt, s = 0.0, 0, 0j
    for k in range(chunks(x)):
        c = np.asarray(x[k * C.N:(k + 1) * C.N], dtype=np.complex64); s += c.sum(dtype=np.complex128); cnt += len(c)
    m = s / cnt
    for k in range(chunks(x)):
        c = np.asarray(x[k * C.N:(k + 1) * C.N], dtype=np.complex64); tot += float(np.sum(np.abs(c - m) ** 2, dtype=np.float64))
    return tot / cnt


def noise_power_self(x):
    """기준 데이터가 없을 때: 조각마다 평균 전력의 10배 넘는 표본을 빼고 잰 평균 전력의 중앙값."""
    v = []
    for k in range(chunks(x)):
        c = np.asarray(x[k * C.N:(k + 1) * C.N], dtype=np.complex64); p = np.abs(c - c.mean()) ** 2
        m = p.mean()
        for _ in range(2): m = p[p < 10 * m].mean()
        v.append(m)
    return float(np.median(v))


def iq_to_snapshots(x, noise):
    n = chunks(x); out = np.empty((n, C.NF, C.NT), np.uint8); g = np.float32(1 / np.sqrt(noise))
    for k in range(n):
        out[k] = C.to_u8(C.spec_db(np.asarray(x[k * C.N:(k + 1) * C.N], dtype=np.complex64) * g))
    return out


# ---------------- 판정 ----------------
def alarms(flags, k, n):
    w = np.convolve(flags.astype(int), np.ones(n, int))[: len(flags)]
    return w >= k


def episodes(a):
    """경보가 이어진 범위 [(시작 번호, 끝 번호)]."""
    out, s = [], None
    for i, v in enumerate(a):
        if v and s is None: s = i
        if not v and s is not None: out.append((s, i - 1)); s = None
    if s is not None: out.append((s, len(a) - 1))
    return out


def report(s, thr, name, args):
    """판정, 경보, CSV 저장, 화면 출력."""
    flags = s > thr; a = alarms(flags, args.k, args.n)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("index,time_s,score,threshold,anomaly,alarm\n")
        for i in range(len(s)): f.write(f"{i},{i * args.interval:.3f},{s[i]:.5f},{thr:.5f},{int(flags[i])},{int(a[i])}\n")
    ep = episodes(a)
    print(f"{name}, 판정 임계값 {thr:.4f}(검증셋 정상 이상 점수의 상위 {args.fpr * 100:g} % 값), 스냅샷 {len(s)}장 중 이상 판정 {int(flags.sum())}장")
    print(f"경보(최근 {args.n}장 중 {args.k}장 이상): {len(ep)}번" + "".join(f"\n  {a0 * args.interval:g}–{a1 * args.interval:g}초(스냅샷 {a0}–{a1})" for a0, a1 in ep))
    print(f"결과 파일: {out}")
    return s, flags, a


def load_site(d):
    """장소 폴더(src/site_train.py 출력): site.json, pixel_stats.npz, cae<크기>_s<시드>.pt"""
    d = Path(d)
    if not (d / "site.json").exists(): raise SystemExit(f"{d}: site.json 이 없음. 먼저 python src/site_train.py 로 장소 폴더를 만들 것")
    st = np.load(d / "pixel_stats.npz")
    return json.load(open(d / "site.json", encoding="utf-8")), st["mu"].astype(np.float32), st["sd"].astype(np.float32)


def predict_site(u8, args, d, meta, mu, sd):
    key = "0.001" if args.fpr == 0.001 else "0.01"
    if args.model == "pixel":
        p = meta["pixel"]; s = pixel_scores(u8, mu, sd, p["rows"], p["cols"]); thr = p["thr"][key]
        name = f"장소 {meta['name']}: (나) 정상 평균과의 픽셀 차이"
    else:
        import torch
        from cae import CAE
        k = args.cae or meta["default_cae"]
        if k not in meta["cae"]: raise SystemExit(f"--cae {k} 가 장소 폴더에 없음. 있는 것: {', '.join(meta['cae'])}")
        m = meta["cae"][k]; dev = device(); net = CAE(tuple(m["channels"]))
        net.load_state_dict(torch.load(Path(d) / m["file"], map_location=dev)); net.to(dev).eval()
        s = cae_scores(net, u8, mu, sd, m["rows"], m["cols"], dev); thr = m["thr"][key]
        name = f"장소 {meta['name']}: CAE 파라미터 {m['n_params']:,}개(시드 {m['seed']})"
    return report(s, thr, name, args)


def predict(u8, args):
    mu, sd, meta = load_meta(); key = "0.001" if args.fpr == 0.001 else "0.01"
    if args.model == "cnn":
        dev = device(); s = cnn_scores(load_cnn(args.seed, dev), u8, mu, sd, dev)
        thr = meta["cnn"][str(args.seed)]["thr"][key]; name = f"(다) CNN 시드 {args.seed}"
    elif args.model == "cae":
        dev = device(); m = meta["cae"][str(args.seed)]; s = cae_scores(load_cae(args.seed, dev), u8, mu, sd, m["rows"], m["cols"], dev)
        thr = m["thr"][key]; name = f"(라) CAE 시드 {args.seed}"
    elif args.model == "ens":
        from ensemble import combine
        dev = device(); k = str(args.seed); m = meta["cae"][k]; e = meta["ens"][k]
        sc = cnn_scores(load_cnn(args.seed, dev), u8, mu, sd, dev)
        se = cae_scores(load_cae(args.seed, dev), u8, mu, sd, m["rows"], m["cols"], dev)
        s = combine(sc, se, e); thr = e["thr"][key]; name = f"(마) CNN·CAE 앙상블 시드 {args.seed}"
    else:
        p = meta["pixel"]; s = pixel_scores(u8, mu, sd, p["rows"], p["cols"]); thr = p["thr"][key]; name = "(나) 정상 평균과의 픽셀차"
    return report(s, thr, name, args)


def eval_sets():
    """make_data.py 로 만든 데이터셋의 점수를 저장소의 학습된 모델(CNN 3개, CAE 3개)로 계산해 runs/ 에 저장한다(학습 없이 결과 재현).
    만드는 파일: runs/scores_cnn.npz, runs/cnn_select.json, runs/scores_cae.npz, runs/cae_select.json (evaluate.py 가 읽음).
    threshold 는 검증셋 정상 스냅샷 점수에서 다시 계산하고 models/thresholds.json 의 값과 함께 적는다."""
    mu, sd, meta = load_meta(); dev = device(); C.RUNS.mkdir(exist_ok=True)
    D = {n: np.load(C.DATA / f"{n}.npy", mmap_mode="r") for n in SETS}
    for g, pre in (("cnn", "C"), ("cae", "E")):
        out, sel = {}, {}
        for s in ("0", "1", "2"):
            m = meta[g][s]
            if g == "cnn":
                net = load_cnn(int(s), dev); f = lambda X: cnn_scores(net, X, mu, sd, dev)
            else:
                net = load_cae(int(s), dev); f = lambda X: cae_scores(net, X, mu, sd, m["rows"], m["cols"], dev)
            for n in SETS: out[f"{pre}{s}/{n}"] = f(D[n])
            v = out[f"{pre}{s}/val_normal"]; t1, t01 = float(np.quantile(v, 0.99)), float(np.quantile(v, 0.999))
            sel[s] = dict(history=m["history"], best_epoch=m["best_epoch"], thr_fpr1=t1, thr_fpr01=t01)
            if g == "cae": sel[s].update(rows=m["rows"], cols=m["cols"])
            print(f"{g.upper()} 시드 {s}: 검증셋 정상의 1 %가 넘는 threshold {t1:.4f}(models/ 값 {m['thr']['0.01']:.4f}), 0.1 %가 넘는 threshold {t01:.4f}(models/ 값 {m['thr']['0.001']:.4f})", flush=True)
        np.savez(C.RUNS / f"scores_{g}.npz", **out)
        json.dump(sel, open(C.RUNS / f"{g}_select.json", "w"), ensure_ascii=False, indent=1)
    print("PREDICT_EVAL_SETS_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description="학습된 모델로 이상 탐지 추론")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--iq"); g.add_argument("--snapshots"); g.add_argument("--eval-sets", action="store_true")
    ap.add_argument("--noise-ref"); ap.add_argument("--model", choices=["cnn", "cae", "ens", "pixel"], default=None,
                    help="기본: --site 가 있으면 cae, 없으면 cnn. --site 에서는 cae·pixel 만 쓴다")
    ap.add_argument("--site", help="src/site_train.py 가 만든 장소 폴더"); ap.add_argument("--cae", help="--site 에서 쓸 CAE, 예: 65k_s0")
    ap.add_argument("--seed", type=int, choices=[0, 1, 2], default=0); ap.add_argument("--fpr", type=float, choices=[0.001, 0.01], default=0.001)
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0); ap.add_argument("--out", default=str(C.RUNS / "predict.csv"))
    args = ap.parse_args()
    if args.model is None: args.model = "cae" if args.site else "cnn"
    if args.site and args.model not in ("cae", "pixel"): raise SystemExit("--site 에서는 --model cae 또는 pixel 만 쓴다")
    if args.eval_sets: return eval_sets()
    site = load_site(args.site) if args.site else None
    if args.snapshots:
        u8 = np.load(args.snapshots)
        if u8.dtype != np.uint8 or u8.shape[1:] != (C.NF, C.NT): raise SystemExit(f"스냅샷 형식이 다름: {u8.dtype} {u8.shape}, 필요한 것 uint8 (장수, {C.NF}, {C.NT})")
    else:
        x = read_iq(args.iq)
        if args.noise_ref:
            noise = noise_power_ref(read_iq(args.noise_ref)); print(f"잡음 전력(기준 데이터 {args.noise_ref}): {noise:.4g}")
        elif site:
            noise = site[0]["noise_power"]; print(f"배경 잡음 세기(장소 폴더 {args.site}, 학습 때 잰 값): {noise:.4g}")
        else:
            noise = noise_power_self(x)
            print(f"경고: --noise-ref 가 없어 입력에서 잡음 전력을 추정함({noise:.4g}). 시험 전 같은 장소에서 받은 정상 IQ 데이터를 주는 것을 권장", flush=True)
        u8 = iq_to_snapshots(x, noise)
    if site: return predict_site(u8, args, args.site, *site)
    predict(u8, args)


if __name__ == "__main__":
    main()
