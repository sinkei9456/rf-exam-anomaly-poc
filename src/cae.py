"""(라) 합성곱 오토인코더(CAE) 이상 탐지 (2026-11-18). 정상 스냅샷만으로 학습한다. GPU 1개, 시드 3개를 차례로.

입력: CNN 과 같은 표준점수 지도 z = (스냅샷 dB - 픽셀별 정상 평균) / 픽셀별 정상 표준편차, -4–12 로 자르고 4 로 나눔. 1 × 256 × 97.
모델: 인코더 = 3×3 합성곱(stride 2) 4단, 채널 16·32·64·16. 256×97 → 16×7×16(숫자 1,792개, 입력의 7 %).
      디코더 = 2배 키우기 + 3×3 합성곱 4단, 마지막에 256×97 로 맞춘다.
학습: 학습셋 정상 10,000장만. 손실 = 입력과 복원의 평균제곱오차. 시간축 순환 이동·뒤집기 증강(CNN 과 같음).
      AdamW, 학습률 1e-3(OneCycle), 20 에포크, 배치 크기 128.
에포크 선택: 검증셋 정상 5,000장의 복원 오차(평균제곱오차)가 가장 작은 에포크. 이상 데이터는 쓰지 않는다.
이상 점수: 입력 - 복원의 양수 부분(이상 신호는 에너지를 더한다) → r줄 × c칸 상자 평균(상자가 이미지 안에 들어가는 위치만) → 최댓값.
  평활 크기 후보와 선택 규칙은 (가)(나)와 같다. 검증셋 정상 스트림 점수의 상위 1 %를 threshold 로 두었을 때
  검증셋 사각형 덩어리 2,000장의 recall 이 가장 큰 크기. 사각형 덩어리는 이 선택에만 쓴다.
threshold: 고른 크기에서 검증셋 정상 점수의 FPR 1 %·0.1 %.
출력: runs/cae_s<시드>.pt, runs/scores_cae.npz(키 "E<시드>/<데이터셋>"), runs/cae_select.json
사용: python src/cae.py [--seeds 0 1 2] [--epochs 20]
"""
import sys, json, time, argparse, gc
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C
from cnn import Prep, gpu_u8, dev
from pixeldiff import ROWS, COLS

SETS = ["val_normal", "val_blob", "test_normal", "test_event", "test_demo"]


def down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def up(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(down(1, 16), down(16, 32), down(32, 64), nn.Conv2d(64, 16, 3, stride=2, padding=1))
        self.dec = nn.Sequential(up(16, 64), up(64, 32), up(32, 16), nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, x):
        return F.interpolate(self.dec(self.enc(x)), size=x.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def recon_loss(net, prep, X, bs=512):
    net.eval(); tot = 0.0
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs]); tot += F.mse_loss(net(x), x, reduction="sum").item()
    return tot / (len(X) * C.NF * C.NT)


@torch.no_grad()
def scores_all(net, prep, X, kernels, bs=256):
    """상자 크기마다 (입력 - 복원)+ 의 평활 최댓값. 반환 [len(kernels), len(X)]. avg_pool2d(패딩 없음) = 상자가 이미지 안에 들어가는 위치만."""
    net.eval(); out = torch.empty(len(kernels), len(X))
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs]); r = (x - net(x)).clamp(min=0)
        for j, (rr, cc) in enumerate(kernels):
            out[j, i:i + len(x)] = F.avg_pool2d(r, (rr, cc), stride=1).flatten(1).max(1).values.float().cpu()
    return out.numpy()


def train_one(seed, epochs, prep, Xtr, Xvn, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    net = CAE().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    g = torch.Generator(device=dev); g.manual_seed(seed); ar = torch.arange(C.NT, device=dev)
    hist, best, best_state, best_ep = [], 1e9, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev, generator=g); tl = 0.0; t = time.time()
        for i in range(len(Xtr) // bs):
            idx = perm[i * bs:(i + 1) * bs]; x = prep(Xtr[idx])
            sh = torch.randint(0, C.NT, (len(idx), 1), device=dev, generator=g)
            x = torch.gather(x, 3, ((ar[None, :] + sh) % C.NT)[:, None, None, :].expand_as(x))
            fl = torch.rand(len(idx), device=dev, generator=g) < 0.5; x[fl] = x[fl].flip(3)
            loss = F.mse_loss(net(x), x); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tl += loss.item()
        vl = recon_loss(net, prep, Xvn)
        hist.append(dict(epoch=ep + 1, train_loss=tl / (len(Xtr) // bs), val_loss=vl))
        print(f"시드 {seed} 에포크 {ep + 1:2d}: 학습 복원 오차 {hist[-1]['train_loss']:.5f}, 검증 정상 복원 오차 {vl:.5f}, {time.time() - t:.0f}초", flush=True)
        if vl < best: best, best_ep, best_state = vl, ep + 1, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, hist, best_ep


def main(seeds, epochs):
    prep = Prep(); Xtr = gpu_u8("train_normal"); D = {n: gpu_u8(n) for n in SETS}
    kernels = [(r, c) for r in ROWS for c in COLS]
    f = C.RUNS / "scores_cae.npz"; out = dict(np.load(f)) if f.exists() else {}
    fj = C.RUNS / "cae_select.json"; sel = json.load(open(fj)) if fj.exists() else {}
    for s in seeds:
        net, hist, best_ep = train_one(s, epochs, prep, Xtr, D["val_normal"])
        torch.save(net.state_dict(), C.RUNS / f"cae_s{s}.pt")
        sn, sb = scores_all(net, prep, D["val_normal"], kernels), scores_all(net, prep, D["val_blob"], kernels)
        cand = []
        for j, (r, c) in enumerate(kernels):
            t1 = float(np.quantile(sn[j], 0.99))
            cand.append(dict(rows=r, cols=c, val_recall_fpr1=float((sb[j] > t1).mean()), val_auroc=float((sb[j][:, None] > sn[j][None, :]).mean())))
        b = max(range(len(kernels)), key=lambda j: (cand[j]["val_recall_fpr1"], cand[j]["val_auroc"])); kr = [kernels[b]]
        for n in SETS: out[f"E{s}/{n}"] = scores_all(net, prep, D[n], kr)[0]
        v = out[f"E{s}/val_normal"]
        sel[str(s)] = dict(history=hist, best_epoch=best_ep, rows=kernels[b][0], cols=kernels[b][1], candidates=cand,
                           thr_fpr1=float(np.quantile(v, 0.99)), thr_fpr01=float(np.quantile(v, 0.999)))
        print(f"시드 {s}: 고른 에포크 {best_ep}, 평활 {kernels[b][0]}줄 × {kernels[b][1]}칸(검증 recall {cand[b]['val_recall_fpr1'] * 100:.1f} %), "
              f"threshold FPR1% {sel[str(s)]['thr_fpr1']:.4f}, FPR0.1% {sel[str(s)]['thr_fpr01']:.4f}", flush=True)
        np.savez(f, **out); json.dump(sel, open(fj, "w"), ensure_ascii=False, indent=1)
        del net; gc.collect(); torch.cuda.empty_cache()
    print("CAE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=20)
    A = ap.parse_args(); main(A.seeds, A.epochs)
