"""추론 파이프라인 시험용 IQ 파일 만들기 (2026-11-18). 실제 수신기 대신 쓴다.

data/demo_iq.npy        1초에 한 번 받은 20 ms 조각 60개(1분)를 이어 붙인 IQ 데이터(complex64, 60 × 400,000 표본).
                        20–29번 조각(10초)에만 이상 1(블루투스 이어폰 오디오 전송 흉내, SNR 6 dB)을 넣는다. 나머지는 정상.
data/demo_noise_ref.npy 정상 상태에서 받은 IQ 조각 10개(잡음 세기 기준용, 와이파이·블루투스 없음).
배경 노이즈는 테스트셋과 같은 공원 무신호 2–3초 부분을 쓴다. 실제 수신기처럼 보이게 전체 세기에 임의의 배율(37)을 곱한다.
predict.py 가 잡음 세기를 맞추는지 보려는 것이다.
사용: python src/make_demo_iq.py   →   python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

GAIN = 37.0
N_CHUNK, EVENT = 60, range(20, 30)


def main():
    C.DATA.mkdir(exist_ok=True)
    x = np.empty(N_CHUNK * C.N, np.complex64); n_vis = []
    for k in range(N_CHUNK):
        rng = np.random.default_rng([9, k])
        c, _ = C.bg_snapshot("test", rng); C.add_normal(c, rng)
        if k in EVENT:
            _, _, v = C.add_anomaly(c, rng, 1, 6.0); n_vis.append(v)
        x[k * C.N:(k + 1) * C.N] = c * GAIN
    np.save(C.DATA / "demo_iq.npy", x)
    ref = np.empty(10 * C.N, np.complex64)
    for k in range(10):
        c, _ = C.bg_snapshot("test", np.random.default_rng([8, k])); ref[k * C.N:(k + 1) * C.N] = c * GAIN
    np.save(C.DATA / "demo_noise_ref.npy", ref)
    print(f"data/demo_iq.npy: 조각 {N_CHUNK}개, 이상 1(SNR 6 dB)은 {EVENT.start}–{EVENT.stop - 1}번, 그중 패킷이 보이는 조각 {sum(v > 0 for v in n_vis)}개")
    print("data/demo_noise_ref.npy: 정상 조각 10개")


if __name__ == "__main__":
    main()
