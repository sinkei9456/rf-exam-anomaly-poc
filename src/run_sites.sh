#!/bin/bash
# 장소별 실험 전체 (2026-11-19). 저장소 맨 위 기준 상대 경로만 쓴다.
#
# 사용: bash src/run_sites.sh      (먼저 실내 IQ 데이터를 받아 python src/site_setup.py 로 배경 파일을 만든다. README 의 "데이터 받기")
# 장소(실내, 실내(드론 끔), 공원)마다 차례로:
#   데이터 합성(CPU) → (가)(나) 픽셀 차이 → CAE 3종(파라미터 65,265·21,121·6,489개) × 시드 3개 학습(GPU 하나, 한 번에 하나) → 평가·부록 그림
# 세 장소가 끝나면 장소 바꿔 쓰기(site_cross.py) → 결과 모으기(site_summary.py, runs/site/results.md).
# 끝까지 마친 단계는 다시 하지 않는다(데이터 파일, scores_pixel.npz, cae<크기>_s2.pt 가 있으면 건너뜀).
# 파이썬은 PY 로 바꿀 수 있다: PY=/path/to/python bash src/run_sites.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
for s in indoor indoor_droneoff park; do
  d=runs/site/$s; mkdir -p $d
  echo "[$s] 데이터 합성 $(date +%T)"
  EXAM_SITE=$s OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY src/make_data.py --workers ${WORKERS:-16} > $d/make_data.log 2>&1
  if [ ! -f $d/scores_pixel.npz ]; then
    echo "[$s] (가)(나) 픽셀 차이 $(date +%T)"; EXAM_SITE=$s $PY src/pixeldiff.py > $d/pixeldiff.log 2>&1
  fi
  for z in 65k 21k 6k; do
    if [ ! -f $d/cae${z}_s2.pt ]; then
      echo "[$s] CAE $z 학습 $(date +%T)"; EXAM_SITE=$s $PY src/cae.py --size $z --seeds 0 1 2 --epochs 20 > $d/cae$z.log 2>&1
    fi
  done
  echo "[$s] 평가·부록 그림 $(date +%T)"; EXAM_SITE=$s $PY src/site_evaluate.py > $d/evaluate.log 2>&1
done
echo "[장소 바꿔 쓰기] $(date +%T)"; $PY src/site_cross.py > runs/site/cross.log 2>&1
echo "[결과 모으기] $(date +%T)"; $PY src/site_summary.py > runs/site/summary.log 2>&1
echo "RUN_SITES_DONE $(date +%T)"
