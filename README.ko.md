# rf-exam-anomaly-poc

[English](README.md) | 한국어

시험장에서 몰래 쓰는 무선 기기(블루투스 이어폰, AI 안경)의 신호를 수신기 스펙트로그램에서 찾을 수 있는지, 합성 데이터로 확인한 개념 검증(PoC) 코드입니다. 정상 상태를 배우고 벗어난 것을 찾는 이상 탐지 방법 네 가지를 비교했습니다.

- (가) 직전 스냅샷과의 픽셀차
- (나) 정상 평균과의 픽셀차
- (다) CNN: 정상 스냅샷과, 정상 스냅샷에 사각형 덩어리를 합성한 스냅샷을 구분하도록 학습한 분류 모델
- (라) CAE(합성곱 오토인코더): 정상 스냅샷만 학습하고, 복원하지 못한 부분을 이상으로 보는 모델

방법과 결과 전체는 [doc/00_overview.md](doc/00_overview.md)에 있습니다. 같은 내용을 왼쪽 목차가 붙은 웹 페이지([문서 웹 페이지](https://sinkei9456.github.io/rf-exam-anomaly-poc/doc/00_overview.html))로도 볼 수 있습니다.

## 결과 요약

공원에서 받은 무신호 IQ 데이터 3초를 배경 노이즈로 쓰고, 그 위에 약한 와이파이·블루투스(정상)와 처음 보는 블루투스 2종(이상)을 합성했습니다. 아래는 테스트셋 결과입니다. threshold(이상 점수가 이 값을 넘으면 이상으로 판정하는 경계)는 검증셋 정상 스냅샷 5,000장 중 0.1 %(5장)만 넘도록 정했고, CNN과 CAE는 시드 3개의 평균입니다.

- FPR: 정상 스냅샷을 이상으로 잘못 판정한 비율
- recall: 부정 기기 신호가 보이는 스냅샷을 이상으로 판정한 비율
- SNR: 이상 신호가 잡음보다 몇 dB 센지. 정상 와이파이·블루투스는 SNR 2–10 dB입니다.

**스냅샷 한 장으로 판단할 때 FPR과 recall (%)**

| 방법 | FPR | recall, SNR 0–3 dB | recall, SNR 6 dB | recall, SNR 9 dB | recall, SNR 12 dB |
|---|---:|---:|---:|---:|---:|
| (가) 직전 스냅샷과의 픽셀차 | 0.09 | 0.0–0.6 | 0.0–0.3 | 0.5–37.1 | 46.1–88.8 |
| (나) 정상 평균과의 픽셀차 | 0.11 | 0.0–0.6 | 0.0–0.3 | 2.7–56.5 | 70.9–92.6 |
| (다) CNN | 0.30 | 89.4–95.5 | 90.7–92.8 | 93.5–95.2 | 94.6–96.7 |
| (라) CAE | 0.05 | 0.0–2.2 | 18.4–41.2 | 73.9–76.8 | 85.4–91.9 |

recall의 범위는 이상 신호 2종(이어폰형, BLE형)의 값 중 낮은 값과 높은 값입니다.

- 픽셀차는 이상 신호가 정상 신호보다 셀 때만 찾습니다.
- 정상만 배운 CAE는 픽셀차보다 약 3 dB 약한 신호까지 찾고, 오경보가 가장 적습니다.
- CNN은 SNR과 거의 상관없이 찾습니다. 다만 학습에 쓴 사각형 덩어리가 테스트 신호의 크기를 포함하므로 CAE보다 유리한 조건입니다.
- 수신기가 20 MHz만 봐서 기기가 켜져 있어도 스냅샷의 17–40 %에는 아무것도 안 보입니다. 그래서 "최근 10장 중 2장 이상이면 경보"처럼 10초 단위로 판단합니다. 이 규칙에서 1시간에 울린 오경보는 CNN 0–1.3번, CAE 0번이었습니다.
- 모두 합성 데이터 결과입니다. 실제 시험장 데이터로는 확인하지 않았습니다.

## 저장소 구성

```
rf-exam-anomaly-poc/
├── data/background/   배경 노이즈 IQ 파일(릴리스에서 내려받아 여기에 풂)
├── doc/               전체 정리 문서(00_overview.md, .html), 그림, 선행 조사(prior_work.md)
├── models/            학습된 모델: CNN 3개, CAE 3개, 정상 통계, threshold
├── results/           문서에 적은 기준 결과(재현할 때 비교용)
├── src/               학습 파이프라인, 추론 파이프라인, 평가 코드
├── requirements.txt
└── LICENSE
```

src/ 안의 파일은 doc/00_overview.md 3.5절에 하나씩 적었습니다.

## 설치

Python 3.11에서 확인했습니다. PyTorch는 쓰는 컴퓨터에 맞는 판을 [pytorch.org](https://pytorch.org)의 안내대로 먼저 설치하고, 나머지를 설치합니다.

```
git clone https://github.com/sinkei9456/rf-exam-anomaly-poc.git
cd rf-exam-anomaly-poc
pip install -r requirements.txt
```

학습에는 NVIDIA GPU가 있어야 합니다. 추론은 GPU가 없으면 CPU로 돕니다. 그림 안 글자가 한국어라서 나눔고딕 같은 한글 글꼴이 있으면 좋습니다(없으면 글자만 깨지고 결과는 같습니다).

## 데이터 받기

배경 노이즈 IQ 파일 3개(합계 480 MB)는 GitHub 파일 크기 한도 때문에 저장소가 아니라 릴리스에 올렸습니다. 저장소 맨 위에서 내려받아 풉니다.

```
curl -LO https://github.com/sinkei9456/rf-exam-anomaly-poc/releases/download/v1.0/background_park02.tar
tar xf background_park02.tar        # data/background/Park02_1.npy, Park02_2.npy, Park02_3.npy
cd data/background && sha256sum -c SHA256SUMS.txt && cd ../..
```

파일마다 1초 분량(20 MS/s, complex64)이고, 1번은 학습셋, 2번은 검증셋, 3번은 테스트셋의 배경 노이즈로 씁니다. 원본은 D4D 해커톤 때 공원에서 드론·조종기를 모두 끄고 HackRF One(가운데 주파수 2,449.5 MHz)으로 받은 IQ 데이터입니다. [rf-drone-detector](https://github.com/sinkei9456/rf-drone-detector) 저장소의 릴리스 파일 hackrf_1_train_background.tar에 CC BY 4.0으로 공개되어 있습니다. 그중 3초에서 블루투스 모양 덩어리를 지우고 1초씩 나눈 것이 이 세 파일입니다(같은 저장소의 src/neo2_pieces.py가 만듦).

## 사용법 1: 학습 파이프라인

처음부터 다시 만들려면 아래 한 줄을 실행합니다.

```
bash src/run_all.sh
```

차례로 하는 일은 다음과 같습니다. GPU 1개에서 모두 약 30분 걸립니다.

| 단계 | 스크립트 | 하는 일 | 걸리는 시간 |
|---|---|---|---:|
| 1 | src/make_data.py | 합성 스냅샷 55,000장 만들기(data/, 1.3 GB) | 약 3분 |
| 2 | src/pixeldiff.py | (가)(나) 픽셀차의 평활 크기와 threshold를 검증셋으로 정하고 점수 계산 | 약 1분 |
| 3 | src/cnn.py | (다) CNN 학습, 시드 3개 | 약 20분 |
| 4 | src/cae.py | (라) CAE 학습, 시드 3개 | 약 5분 |
| 5 | src/evaluate.py | 테스트셋 평가, 표(runs/results.md)와 그림(doc/fig/) 만들기 | 약 2분 |
| 6 | src/compare_results.py | 내 결과를 기준 결과(results/)와 비교 | 몇 초 |

학습하지 않고 저장소의 학습된 모델로 문서의 숫자를 그대로 재현하려면 아래처럼 실행합니다. 3·4단계 대신 src/predict.py --eval-sets가 models/의 모델로 점수를 계산합니다. 약 8분 걸립니다.

```
bash src/run_all.sh --use-released-models
```

참고할 점은 다음과 같습니다.

- 파이썬을 바꾸려면 `PY=/경로/python bash src/run_all.sh`처럼 씁니다. 데이터 만들기에 쓰는 CPU 프로세스 수는 `WORKERS`(기본 16)로 바꿉니다.
- 이미 있는 data/<데이터셋>.npy는 다시 만들지 않습니다. 처음부터 다시 하려면 data/의 .npy·.csv·_boxes.json 파일과 runs/를 지웁니다.
- 결과는 runs/에 생깁니다. 다시 학습한 모델을 추론에 쓰려면 `python src/export_models.py`로 runs/의 모델을 models/로 내보냅니다.

## 사용법 2: 추론 파이프라인

src/predict.py는 학습 코드를 돌리지 않고 models/의 파일만으로 추론합니다. 학습된 모델과 추론 코드만 따로 받으려면 릴리스의 rf-exam-anomaly-inference-v1.0.zip(2.6 MB)을 받아 풀면 됩니다. 입력은 HackRF IQ 데이터나 이미 만든 스냅샷이고, 출력은 스냅샷마다의 이상 점수, 이상 판정, 경보입니다.

먼저 시험용 IQ 파일로 돌려 봅니다. 1분짜리 IQ 데이터에서 20–29초에만 이어폰형 이상 신호(SNR 6 dB)가 들어 있습니다.

```
python src/make_demo_iq.py
python src/predict.py --iq data/demo_iq.npy --noise-ref data/demo_noise_ref.npy --model cnn
```

화면에는 다음과 같이 나옵니다.

```
(다) CNN 시드 0, threshold 2.8855(검증셋 정상의 0.1 %가 넘게 정한 값), 스냅샷 60장 중 이상 판정 6장
경보(최근 10장 중 2장 이상): 1번
  22–36초(스냅샷 22–36)
결과 파일: runs/predict.csv
```

경보는 최근 10장 중 2장 이상이 이상일 때 켜지고, 마지막 이상 판정 뒤 10장이 지나면 꺼집니다. 그래서 36초까지 이어집니다.

입력은 셋 중 하나를 고릅니다.

| 입력 옵션 | 내용 |
|---|---|
| `--iq 파일` | IQ 데이터. 1초에 한 번 받은 20 ms 조각(20 MS/s, 400,000 표본)을 시간 순서대로 이어 붙인 파일입니다. 형식은 .npy(복소수), .cf32(complex64), .cs8 또는 .sigmf-data(HackRF int8)입니다. |
| `--snapshots 파일` | 이미 만든 스냅샷(uint8, 장수 × 256 × 97). src/make_data.py가 만드는 형식과 같습니다. |
| `--eval-sets` | src/make_data.py로 만든 검증셋·테스트셋의 점수를 계산해 runs/에 저장합니다. 학습 파이프라인의 --use-released-models가 씁니다. |

나머지 옵션은 다음과 같습니다.

| 옵션 | 기본값 | 내용 |
|---|---|---|
| `--model` | cnn | cnn은 (다) CNN, cae는 (라) CAE, pixel은 (나) 정상 평균과의 픽셀차 |
| `--seed` | 0 | CNN·CAE의 시드(0, 1, 2) |
| `--fpr` | 0.001 | 쓸 threshold. 0.001이면 검증셋 정상 스냅샷의 0.1 %가 넘게 정한 값, 0.01이면 1 %가 넘게 정한 값입니다 |
| `--noise-ref 파일` | 없음 | 잡음 세기를 맞출 기준 IQ 데이터. 같은 장소·같은 수신 설정에서 받은 정상 데이터를 권합니다 |
| `--k`, `--n` | 2, 10 | 최근 n장 중 k장 이상이 이상이면 경보 |
| `--interval` | 1.0 | 스냅샷 사이 간격(초). 결과 파일의 시각 칸에만 씁니다 |
| `--out` | runs/predict.csv | 결과 파일 |

결과 파일(CSV)의 칸은 index(스냅샷 번호), time_s(시각, 초), score(이상 점수), threshold, anomaly(이상 판정, 0 또는 1), alarm(경보, 0 또는 1)입니다.

직접 받은 IQ 데이터에 쓸 때는 다음을 지켜야 합니다.

- 수신 설정이 학습과 같아야 합니다. HackRF, 가운데 주파수 2,449.5 MHz, 20 MS/s이고 이득은 고정합니다.
- 모델은 잡음 세기를 1로 맞춘 입력을 배웠습니다. `--noise-ref`로 정상 상태의 IQ 데이터를 주면 그 평균 전력으로 맞춥니다. 주지 않으면 입력에서 센 표본을 빼고 추정하는데, 신호가 많으면 잡음을 크게 잡습니다(시험용 IQ에서 0.4 dB).
- threshold는 공원 배경으로 정한 값입니다. 장소가 바뀌면 FPR이 검증셋에서 정한 비율과 달라질 수 있습니다. CNN은 검증셋 정상의 0.1 %가 넘게 정했는데, 테스트셋에서는 0.01–0.65 %가 넘었습니다. 새 장소에서는 정상 데이터의 점수를 먼저 보고 threshold를 다시 정하는 것을 권합니다.

파이썬 코드 안에서 쓸 수도 있습니다.

```python
import sys; sys.path.insert(0, "src")
import numpy as np, predict as P
mu, sd, meta = P.load_meta(); dev = P.device()
net = P.load_cnn(0, dev)
u8 = np.load("data/test_demo.npy")                      # 스냅샷(uint8, 장수 × 256 × 97)
score = P.cnn_scores(net, u8, mu, sd, dev)
is_anomaly = score > meta["cnn"]["0"]["thr"]["0.001"]
```

## 재현 확인

이 저장소를 새로 받은 상태에서 두 경로를 돌려 기준 결과(results/)와 비교했습니다. 장비는 NVIDIA DGX Spark(GB10, ARM CPU), Python 3.11, PyTorch 2.14(CUDA 13)입니다.

- 학습된 모델로 재현(`--use-released-models`): 8개 모델 × 52개 지표가 모두 같았습니다. 합성 데이터와 픽셀차 결과도 같았습니다.
- 처음부터 다시 학습(`bash src/run_all.sh`): 픽셀차 결과는 같았고, CNN·CAE는 같은 장비에서도 숫자가 조금 달라졌습니다. 시드 하나씩 보면 지표 하나가 CNN에서 최대 12.7 %p, CAE에서 최대 2.7 %p 달랐습니다. CNN 시드 1이 고른 에포크가 11에서 16으로 바뀐 탓입니다.
- 문서에 쓰는 시드 3개 평균으로 보면 차이가 작습니다. recall은 CNN이 최대 2.9 %p, CAE가 최대 1.2 %p 달랐습니다. 이벤트 recall은 CNN이 모두 같았고 CAE가 최대 1.7 %p 달랐습니다.
- FPR은 CNN이 0.30 %에서 0.33 %로, CAE가 0.05 %에서 0.06 %로 바뀌었습니다. 1시간 오경보(10장 중 2장 규칙)는 CNN이 0.5번에서 0.8번(시드별 최대 1.8번)이 됐고, CAE는 0번 그대로였습니다. 결론은 같습니다.

PyTorch는 GPU 학습을 시드만으로 똑같이 되풀이한다고 보장하지 않습니다. 다른 GPU나 패키지 버전에서는 CNN·CAE 숫자가 조금 달라질 수 있습니다. 결론이 같은지는 src/compare_results.py로 확인합니다.

## 한계

- 정상 블루투스를 모두 짧은 패킷(0.4 ms 이하)으로 합성했습니다. 실제 교실에 긴 패킷을 보내는 허용 기기(의료기기 등)나 옆 교실 기기가 있으면 결과가 달라질 수 있습니다.
- 이상 신호 2종은 블루투스 규격 숫자로 만든 흉내이고, 실제 기기 신호가 아닙니다.
- 배경 노이즈는 한 장소 3초뿐입니다.
- 좌석 위치 추정, 사람 몸에 의한 신호 약화는 다루지 않았습니다.

## 라이선스와 출처

- 코드: MIT 라이선스(LICENSE)
- 배경 노이즈 IQ 데이터, 문서와 그림: CC BY 4.0
- 배경 노이즈 원본: rf-drone-detector, https://doi.org/10.5281/zenodo.23656790
