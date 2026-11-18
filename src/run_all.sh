#!/bin/bash
# 학습 파이프라인: 처음부터 다시 만들기 (2026-11-18). 저장소 맨 위 기준 상대 경로만 쓴다.
#
# 사용:
#   bash src/run_all.sh                        전체. 데이터 → 픽셀차 → CNN 학습 → CAE 학습 → 평가·그림 → 기준 결과와 비교
#                                              (GPU 1개 기준 약 30분: 데이터 3분, 픽셀차 1분, CNN 20분, CAE 5분, 평가 2분)
#   bash src/run_all.sh --use-released-models  CNN·CAE 를 학습하지 않고 저장소의 학습된 모델(models/)로 점수 계산(약 8분). 문서 숫자를 그대로 재현할 때
#
# 먼저 data/background/ 에 배경 IQ 파일 3개(Park02_1.npy, Park02_2.npy, Park02_3.npy)가 있어야 한다(README 의 "데이터 받기").
# 이미 있는 data/<이름>.npy 는 다시 만들지 않는다. 처음부터 다시 하려면 data/*.npy, data/*.csv, data/*_boxes.json, runs/ 를 지운다.
# 파이썬은 PY 로 바꿀 수 있다: PY=/path/to/python bash src/run_all.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p runs
for f in Park02_1.npy Park02_2.npy Park02_3.npy; do
  [ -f "data/background/$f" ] || { echo "data/background/$f 가 없음. README 의 '데이터 받기'를 먼저 할 것"; exit 1; }
done
echo "[1/5] 합성 스냅샷 만들기 (CPU)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY src/make_data.py --workers ${WORKERS:-16} > runs/make_data.log 2>&1
echo "[2/5] (가)(나) 픽셀차";           $PY src/pixeldiff.py > runs/pixeldiff.log 2>&1
if [ "$1" = "--use-released-models" ]; then
  echo "[3/5] 저장소의 학습된 모델(models/)로 CNN·CAE 점수 계산"; $PY src/predict.py --eval-sets > runs/predict_eval.log 2>&1
  echo "[4/5] (학습 건너뜀)"
else
  echo "[3/5] (다) CNN 학습, 시드 3개";  $PY src/cnn.py --seeds 0 1 2 --epochs 20 > runs/cnn.log 2>&1
  echo "[4/5] (라) CAE 학습, 시드 3개";  $PY src/cae.py --seeds 0 1 2 --epochs 20 > runs/cae.log 2>&1
fi
echo "[5/5] 평가·그림";                  $PY src/evaluate.py > runs/evaluate.log 2>&1
$PY src/compare_results.py | tee runs/compare.txt
echo "RUN_ALL_DONE"
