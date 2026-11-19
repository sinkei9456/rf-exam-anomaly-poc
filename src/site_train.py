"""새 장소 학습: 그 장소에서 받은 정상 IQ 데이터만으로 모델을 학습하고 판정 임계값까지 정해 장소 폴더 하나에 저장한다 (2026-11-19).
추론은 python src/predict.py --site <장소 폴더> --iq <새 IQ 파일> 로 한다. 두 번째 실험(src/run_sites.sh)과 같은 전처리·학습·임계값 규칙을 쓰되,
합성 정상 신호는 넣지 않는다(실제 장소 데이터에 이미 그 장소의 정상 신호가 들어 있으므로).

입력: --normal FILE [FILE ...]  그 장소의 정상 IQ 데이터(기기 없음). 받은 순서대로 준다. 형식은 predict.py 와 같다
        (.npy 복소수, .cf32, .cs8 / .sigmf-data HackRF int8). 수신 설정(중심 주파수, 20 MS/s, 이득)은 추론 때와 같아야 한다.
      --layout chunks      (기본) 1초에 한 번 받은 20 ms 조각을 이어 붙인 파일. 조각 하나가 스냅샷 하나다(predict.py 입력과 같은 형식).
      --layout continuous  끊김 없이 받은 파일(예: 1초 파일). 파일 안에서 20 ms 를 무작위로 잘라 스냅샷을 만든다(두 번째 실험과 같은 방식).
처리:
  1. 시간 나누기: 파일들을 이어 놓은 시간의 앞 (1 - val_frac) 은 학습셋, 뒤 val_frac 는 검증셋(기본 25 %). 20 ms 는 파일·셋 경계를 넘지 않는다.
  2. 배경 잡음 세기: 학습셋 구간마다 |x - 평균|² 의 중앙값 ÷ ln 2 를 구해 평균(site_setup.py 와 같은 계산). 모든 데이터를 이 값으로 나눈다.
  3. 스냅샷: 20 ms → 스펙트로그램 256 × 97 → 배경 잡음 세기 대비 dB → uint8(common.py). 검증셋에는 커널 크기를 고르는 데만 쓰는
     모양 없는 사각형 패턴을 넣은 스냅샷도 만든다(두 번째 실험과 같음, 기기 신호 모양은 쓰지 않음).
  4. (나) 정상 평균과의 픽셀 차이: 학습셋의 픽셀별 평균·표준편차, 커널 크기, 판정 임계값.
  5. CAE: 크기(--sizes 65k 21k 6k 중), 시드(--seeds)마다 학습. 에포크는 검증셋 정상 재구성 오차로, 커널 크기는 검증셋 사각형 패턴으로 고른다.
  6. 판정 임계값: 검증셋 정상 이상 점수의 상위 1 % 값과 상위 0.1 % 값. 검증 스냅샷이 1,000장보다 적으면 상위 0.1 % 값은 믿기 어렵다고 경고한다.
출력(--out 폴더): site.json(배경 잡음 세기, 커널, 임계값, 학습 기록, 입력 파일), pixel_stats.npz, cae<크기>_s<시드>.pt
사용 예(실내 1초 파일 1–4로 학습):
  python src/site_train.py --layout continuous --out sites/indoor \\
      --normal data/raw/hackrf/1_nodrone_indoor/20261007_antenna_on/recordings/0{1,2,3,4}/*.cs8
"""
import sys, json, time, argparse, gc
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from predict import open_iq
from pixeldiff import ROWS, COLS, score_maps

_FILES = []                                                     # 자식 프로세스가 물려받는 IQ 파일(mmap)
_NOISE = [1.0]


def segments(files, val_frac):
    """파일들을 이어 놓은 시간의 앞부분 = 학습셋, 뒷부분 = 검증셋. 반환 {"train": [(파일 번호, 시작, 끝)], "val": [...]}."""
    lens = [len(a) for a in files]; tot = sum(lens); cut = int(round(tot * (1 - val_frac))); seg = {"train": [], "val": []}; pos = 0
    for k, n in enumerate(lens):
        a, b = pos, pos + n
        if b <= cut: seg["train"].append((k, 0, n))
        elif a >= cut: seg["val"].append((k, 0, n))
        else: seg["train"].append((k, 0, cut - a)); seg["val"].append((k, cut - a, n))
        pos = b
    for r in seg: seg[r] = [s for s in seg[r] if s[2] - s[1] >= C.N]
    if not seg["train"] or not seg["val"]: raise SystemExit("학습셋이나 검증셋에 20 ms 이상인 구간이 없음. 정상 데이터를 더 길게 주거나 --val-frac 를 바꿀 것")
    return seg


def noise_power(files, segs):
    v = []
    for k, a, b in segs:
        x = np.asarray(files[k][a:b], dtype=np.complex64); x = x - x.mean()
        v.append(float(np.median(x.real ** 2 + x.imag ** 2) / np.log(2)))
    return float(np.mean(v)), v


def windows(segs, layout, n, rng):
    """스냅샷을 자를 위치 목록 [(파일 번호, 시작 표본)]. chunks = 구간 안의 연속 20 ms 조각 전부, continuous = 무작위 n 개."""
    if layout == "chunks":
        return [(k, a + j * C.N) for k, a, b in segs for j in range((b - a) // C.N)]
    lens = [b - a - C.N + 1 for k, a, b in segs]; tot = sum(lens); out = []
    for o in rng.integers(0, tot, n):
        j = 0
        while o >= lens[j]: o -= lens[j]; j += 1
        out.append((segs[j][0], segs[j][1] + int(o)))
    return out


def _snap(task):
    k, off, blob_seed = task
    x = np.asarray(_FILES[k][off: off + C.N], dtype=np.complex64) / np.float32(np.sqrt(_NOISE[0]))
    if blob_seed is not None: C.add_blobs(x, np.random.default_rng(blob_seed))
    return C.to_u8(C.spec_db(x))


def make_snapshots(tasks, workers):
    with Pool(workers) as P: return np.stack(P.map(_snap, tasks, chunksize=32))


def thresholds(v):
    return {"0.01": float(np.quantile(v, 0.99)), "0.001": float(np.quantile(v, 0.999))}


def pick_kernel(score_fn):
    """커널 후보마다 검증셋 정상 이상 점수의 상위 1 % 값을 임계값으로 두고, 사각형 패턴을 가장 많이 잡는 크기(같으면 AUROC 큰 것)."""
    best = None
    for r in ROWS:
        for c in COLS:
            sn, sb = score_fn(r, c); t = np.quantile(sn, 0.99)
            key = (float((sb > t).mean()), float((sb[:, None] > sn[None, :]).mean()))
            if best is None or key > best[0]: best = (key, r, c)
    return best[1], best[2], best[0][0]


def train_cae(size, seed, epochs, Xtr, Xvn, Xvb, prep, dev, steps_per_epoch, bs=128):
    import torch, torch.nn.functional as F
    from cae import CAE, SIZES, recon_loss, scores_all
    torch.manual_seed(seed); np.random.seed(seed)
    net = CAE(SIZES[size]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps_per_epoch)
    g = torch.Generator(device=dev); g.manual_seed(seed); ar = torch.arange(C.NT, device=dev)
    hist, best, best_state, best_ep = [], 1e9, None, 0; t0 = time.time()
    for ep in range(epochs):
        net.train(); tl = 0.0; t = time.time()
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(Xtr), (bs,), device=dev, generator=g); x = prep(Xtr[idx])      # 데이터가 적어도 같은 학습량이 되게 복원 추출
            sh = torch.randint(0, C.NT, (bs, 1), device=dev, generator=g)
            x = torch.gather(x, 3, ((ar[None, :] + sh) % C.NT)[:, None, None, :].expand_as(x))
            fl = torch.rand(bs, device=dev, generator=g) < 0.5; x[fl] = x[fl].flip(3)
            loss = F.mse_loss(net(x), x); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tl += loss.item()
        vl = recon_loss(net, prep, Xvn); hist.append(dict(epoch=ep + 1, train_loss=tl / steps_per_epoch, val_loss=vl, seconds=time.time() - t))
        print(f"  CAE {size} 시드 {seed} 에포크 {ep + 1:2d}: 학습 재구성 오차 {hist[-1]['train_loss']:.5f}, 검증 정상 재구성 오차 {vl:.5f}", flush=True)
        if vl < best: best, best_ep, best_state = vl, ep + 1, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); train_sec = time.time() - t0
    kernels = [(r, c) for r in ROWS for c in COLS]
    sn_all, sb_all = scores_all(net, prep, Xvn, kernels), scores_all(net, prep, Xvb, kernels)
    r, c, rec = pick_kernel(lambda r, c: (sn_all[kernels.index((r, c))], sb_all[kernels.index((r, c))]))
    v = sn_all[kernels.index((r, c))]
    info = dict(size=size, seed=seed, channels=list(SIZES[size]), n_params=int(sum(p.numel() for p in net.parameters())), rows=r, cols=c,
                thr=thresholds(v), best_epoch=best_ep, history=hist, train_seconds=train_sec, val_rect_recall_top1=rec, file=f"cae{size}_s{seed}.pt")
    return net, info


def main(a):
    import torch
    from cnn import Prep
    t_start = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = [open_iq(f) for f in a.normal]; _FILES.extend(files)
    seg = segments(files, a.val_frac)
    nf, per = noise_power(files, seg["train"]); _NOISE[0] = nf
    sec = {r: sum(b - s for _, s, b in seg[r]) / C.FS for r in seg}
    print(f"정상 데이터 {len(files)}개 파일, 학습셋 {sec['train']:.1f}초, 검증셋 {sec['val']:.1f}초, 배경 잡음 세기 {nf:.4g}(입력 단위)", flush=True)
    rng = np.random.default_rng(a.seed_data)
    wt, wv = windows(seg["train"], a.layout, a.n_train, rng), windows(seg["val"], a.layout, a.n_val, rng)
    if len(wv) < 1000: print(f"경고: 검증 스냅샷이 {len(wv)}장뿐이라 상위 0.1 % 값(판정 임계값)은 믿기 어렵다. 정상 데이터를 더 길게(1초에 한 장이면 17분 이상) 받는 것을 권장", flush=True)
    wb = [wv[int(i)] for i in rng.integers(0, len(wv), a.n_val_blob)]
    t = time.time()
    Xtr = make_snapshots([(k, o, None) for k, o in wt], a.workers)
    Xvn = make_snapshots([(k, o, None) for k, o in wv], a.workers)
    Xvb = make_snapshots([(k, o, 10 ** 6 + i) for i, (k, o) in enumerate(wb)], a.workers)
    print(f"스냅샷: 학습셋 {len(Xtr)}장, 검증셋 정상 {len(Xvn)}장, 검증셋 사각형 패턴 {len(Xvb)}장 ({time.time() - t:.0f}초)", flush=True)
    # (나) 정상 평균과의 픽셀 차이
    Dtr = C.to_db(Xtr); mu = Dtr.mean(0).astype(np.float32); sd = np.maximum(Dtr.std(0), 0.3).astype(np.float32); del Dtr
    np.savez(out / "pixel_stats.npz", mu=mu, sd=sd)
    Zn, Zb = (C.to_db(Xvn) - mu) / sd, (C.to_db(Xvb) - mu) / sd
    pr, pc, prec = pick_kernel(lambda r, c: (score_maps(Zn, r, c)[0], score_maps(Zb, r, c)[0]))
    pix = dict(rows=pr, cols=pc, thr=thresholds(score_maps(Zn, pr, pc)[0]), val_rect_recall_top1=prec); del Zn, Zb
    print(f"(나) 픽셀 차이: 커널 {pr}줄 × {pc}칸, 판정 임계값(상위 0.1 % 값) {pix['thr']['0.001']:.3f}", flush=True)
    # CAE
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = Prep.__new__(Prep); prep.mu = torch.tensor(mu, device=dev); prep.sd = torch.tensor(sd, device=dev)
    gpu = lambda X: torch.from_numpy(X).to(dev)
    Gtr, Gvn, Gvb = gpu(Xtr), gpu(Xvn), gpu(Xvb); steps = max(len(Xtr), a.n_train) // 128
    cae = {}
    for size in a.sizes:
        for seed in a.seeds:
            net, info = train_cae(size, seed, a.epochs, Gtr, Gvn, Gvb, prep, dev, steps)
            torch.save(net.state_dict(), out / info["file"]); cae[f"{size}_s{seed}"] = info
            print(f"CAE {size} 시드 {seed}: 에포크 {info['best_epoch']}, 커널 {info['rows']}줄 × {info['cols']}칸, 판정 임계값(상위 0.1 % 값) {info['thr']['0.001']:.4f}, 학습 {info['train_seconds']:.0f}초", flush=True)
            del net; gc.collect()
            if dev.type == "cuda": torch.cuda.empty_cache()
    meta = dict(
        created=datetime.now().isoformat(timespec="seconds"), name=out.name,
        input=dict(fs_hz=C.FS, snapshot_s=C.WIN, center_mhz=a.center_mhz, layout=a.layout,
                   image="uint8 256 x 97, 0 = -6 dB, 255 = +34 dB, 0 dB = background noise level"),
        normal_files=[Path(f).name for f in a.normal], train_s=sec["train"], val_s=sec["val"],
        noise_power=nf, noise_power_per_segment=per, noise_units="predict.py 의 open_iq 로 읽은 값(.cs8·.sigmf-data 는 int8 ÷ 128)",
        n_snapshots=dict(train=len(Xtr), val_normal=len(Xvn), val_rect=len(Xvb)),
        pixel=pix, cae=cae, default_cae=f"{a.sizes[0]}_s{a.seeds[0]}", total_seconds=time.time() - t_start)
    json.dump(meta, open(out / "site.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"장소 폴더: {out}  (모두 {meta['total_seconds']:.0f}초)\n추론: python src/predict.py --site {out} --iq <새 IQ 파일>", flush=True)
    print("SITE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="새 장소의 정상 IQ 데이터로 학습하고 판정 임계값을 정해 장소 폴더에 저장")
    ap.add_argument("--normal", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["chunks", "continuous"], default="chunks")
    ap.add_argument("--sizes", nargs="+", choices=["65k", "21k", "6k"], default=["65k"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--n-train", type=int, default=10000, help="continuous 일 때 학습 스냅샷 수, chunks 일 때는 에포크당 학습량")
    ap.add_argument("--n-val", type=int, default=5000, help="continuous 일 때 검증 스냅샷 수")
    ap.add_argument("--n-val-blob", type=int, default=2000); ap.add_argument("--seed-data", type=int, default=0)
    ap.add_argument("--center-mhz", type=float, default=None, help="기록용 중심 주파수(MHz)")
    ap.add_argument("--workers", type=int, default=16)
    main(ap.parse_args())
