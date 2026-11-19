"""장소 학습·추론 시험용 IQ 파일 만들기 (2026-11-19). 실제로 받은 정상 IQ 파일에 이상 신호를 넣어, 1초에 한 번 받은 20 ms 조각 60개(1분)를 만든다.

  python src/make_site_demo_iq.py --background <정상 IQ 파일> --out data/site_demo_iq.npy [--kind 1] [--snr 9] [--start 20] [--len 10]
  - 배경 파일 안에서 무작위 위치의 20 ms 를 60번 잘라 이어 붙인다(합성 정상 신호는 넣지 않음, 받은 그대로).
  - --start 번째 조각부터 --len 개 조각에만 이상 신호를 넣는다(--kind 1 = 블루투스 이어폰 오디오 전송, 2 = BLE 연속 데이터 전송).
    SNR 은 이 파일의 배경 잡음 세기(|x|² 중앙값 ÷ ln 2) 기준이다.
  - 출력 값의 단위는 predict.py 가 배경 파일을 읽은 단위와 같다(.cs8 이면 int8 ÷ 128). 그래서 같은 장소 폴더로 바로 추론할 수 있다.
배경 파일은 학습에 쓰지 않은 파일을 써야 한다(예: 실내 파일 1–4로 학습했으면 파일 5).
"""
import sys, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from predict import open_iq


def main(a):
    x = open_iq(a.background); y = np.asarray(x[0:len(x)], dtype=np.complex64); y = y - y.mean()
    nf = float(np.median(y.real ** 2 + y.imag ** 2) / np.log(2)); g = np.float32(np.sqrt(nf))
    rng = np.random.default_rng(a.seed); out = np.empty(a.chunks * C.N, np.complex64); vis = []
    for k in range(a.chunks):
        o = int(rng.integers(0, len(x) - C.N + 1)); c = np.asarray(x[o:o + C.N], dtype=np.complex64) / g
        if a.start <= k < a.start + a.len:
            _, _, v = C.add_anomaly(c, rng, a.kind, a.snr); vis.append(v)
        out[k * C.N:(k + 1) * C.N] = c * g
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.save(a.out, out)
    print(f"{a.out}: 조각 {a.chunks}개(1초에 한 번 받은 20 ms 가정), 이상 {a.kind}(SNR {a.snr:g} dB)는 {a.start}–{a.start + a.len - 1}번, "
          f"그중 패킷이 보이는 조각 {sum(v > 0 for v in vis)}개, 배경 잡음 세기 {nf:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--background", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kind", type=int, choices=[1, 2], default=1); ap.add_argument("--snr", type=float, default=9.0)
    ap.add_argument("--start", type=int, default=20); ap.add_argument("--len", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
