"""작은 CNN 이상 탐지 (2026-11-18). GPU 1개, 시드 3개를 차례로 학습.

입력: 정상 평균과의 픽셀차 지도 z = (스냅샷 dB - 픽셀별 정상 평균) / 픽셀별 정상 표준편차 (pixeldiff.py 와 같은 값, 학습셋 정상 10,000장으로 계산),
      -4–12 로 자르고 4 로 나눔. 크기 1 × 256 × 97.
학습: 학습셋 정상 10,000장(정답 0) 대 학습셋 사각형 덩어리 10,000장(정답 1). 테스트용 블루투스 2종은 학습에 없다.
  증강: 시간축 무작위 순환 이동, 시간축 뒤집기(스냅샷 안에서 신호가 언제 나오는지는 뜻이 없으므로).
모델: 3×3 합성곱 2개 묶음 4단(채널 16·32·64·64, 사이에 2×2 최대 풀링) → 전체 최대 풀링 → 선형 1개. 출력 = 이상 점수(로짓).
에포크 선택: 검증셋(정상 5,000장의 상위 1 % 점수를 threshold 로 두고 사각형 덩어리 2,000장의 recall)이 가장 높은 에포크. 테스트셋은 보지 않는다.
threshold: 고른 에포크의 검증셋 정상 점수에서 FPR 1 %·0.1 % 가 되는 값.
출력: runs/cnn_s<시드>.pt, runs/scores_cnn.npz(키 "C<시드>/<데이터셋>"), runs/cnn_select.json(에포크별 검증 기록, threshold)
사용: python src/cnn.py [--seeds 0 1 2] [--epochs 20]
"""
import sys, json, time, argparse, gc
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C

SETS = ["val_normal", "val_blob", "test_normal", "test_event", "test_demo"]
dev = torch.device("cuda")


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def make_net():
    return nn.Sequential(block(1, 16), nn.MaxPool2d(2), block(16, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                         block(64, 64), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(64, 1))


class Prep:
    """uint8 이미지(GPU) → z 지도."""
    def __init__(self):
        st = np.load(C.RUNS / "pixel_stats.npz"); self.mu = torch.tensor(st["mu"], device=dev); self.sd = torch.tensor(st["sd"], device=dev)

    def __call__(self, u8):
        db = u8.float() / 255 * (C.DB_HI - C.DB_LO) + C.DB_LO
        return (((db - self.mu) / self.sd).clamp(-4, 12) / 4).unsqueeze(1)


def gpu_u8(name):
    return torch.from_numpy(np.load(C.DATA / f"{name}.npy")).to(dev)


@torch.no_grad()
def scores(net, prep, X, bs=1024):
    net.eval(); return torch.cat([net(prep(X[i:i + bs])).squeeze(1) for i in range(0, len(X), bs)]).float().cpu().numpy()


def val_metrics(sn, sb):
    t1 = float(np.quantile(sn, 0.99)); return float((sb > t1).mean()), float((sb[:, None] > sn[None, :]).mean())


def train_one(seed, epochs, prep, Xtr, ytr, Xvn, Xvb, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss(); g = torch.Generator(device=dev); g.manual_seed(seed)
    hist, best, best_state = [], (-1, -1), None
    ar = torch.arange(C.NT, device=dev)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev, generator=g); tl = 0.0; t = time.time()
        for i in range(len(Xtr) // bs):
            idx = perm[i * bs:(i + 1) * bs]; x = prep(Xtr[idx]); y = ytr[idx]
            sh = torch.randint(0, C.NT, (len(idx), 1), device=dev, generator=g)
            x = torch.gather(x, 3, ((ar[None, :] + sh) % C.NT)[:, None, None, :].expand_as(x))
            fl = torch.rand(len(idx), device=dev, generator=g) < 0.5; x[fl] = x[fl].flip(3)
            loss = lossf(net(x).squeeze(1), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tl += loss.item()
        sn, sb = scores(net, prep, Xvn), scores(net, prep, Xvb); rec, auc = val_metrics(sn, sb)
        hist.append(dict(epoch=ep + 1, train_loss=tl / (len(Xtr) // bs), val_recall_fpr1=rec, val_auroc=auc))
        print(f"시드 {seed} 에포크 {ep + 1:2d}: 학습 손실 {hist[-1]['train_loss']:.4f}, 검증 recall@FPR1% {rec:.3f}, AUROC {auc:.4f}, {time.time() - t:.0f}초", flush=True)
        if (rec, auc) > best: best = (rec, auc); best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}; best_ep = ep + 1
    net.load_state_dict(best_state)
    return net, hist, best_ep


def main(seeds, epochs):
    prep = Prep()
    Xtr = torch.cat([gpu_u8("train_normal"), gpu_u8("train_blob")])
    ytr = torch.cat([torch.zeros(10000, device=dev), torch.ones(10000, device=dev)])
    D = {n: gpu_u8(n) for n in SETS}
    f = C.RUNS / "scores_cnn.npz"; out = dict(np.load(f)) if f.exists() else {}
    fj = C.RUNS / "cnn_select.json"; sel = json.load(open(fj)) if fj.exists() else {}
    for s in seeds:
        net, hist, best_ep = train_one(s, epochs, prep, Xtr, ytr, D["val_normal"], D["val_blob"])
        torch.save(net.state_dict(), C.RUNS / f"cnn_s{s}.pt")
        for n in SETS: out[f"C{s}/{n}"] = scores(net, prep, D[n])
        sn = out[f"C{s}/val_normal"]
        sel[str(s)] = dict(history=hist, best_epoch=best_ep, thr_fpr1=float(np.quantile(sn, 0.99)), thr_fpr01=float(np.quantile(sn, 0.999)))
        print(f"시드 {s}: 고른 에포크 {best_ep}, threshold FPR1% {sel[str(s)]['thr_fpr1']:.3f}, FPR0.1% {sel[str(s)]['thr_fpr01']:.3f}", flush=True)
        np.savez(f, **out); json.dump(sel, open(fj, "w"), ensure_ascii=False, indent=1)
        del net; gc.collect(); torch.cuda.empty_cache()
    print("CNN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=20)
    A = ap.parse_args(); main(A.seeds, A.epochs)
