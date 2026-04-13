#!/bin/bash
set -euo pipefail

cd /data/hejianbiao2023/my_docker_workspace/创新实践2/MixLinear/MixLinear || exit 1

MODEL="WaveMix"
DATA="ETTh1"
ROOT_PATH="${ROOT_PATH:-./dataset}"
DATA_PATH="ETTh1.csv"
CHECKPOINTS="./checkpoints"
GPU="${GPU:-0}"
SWT_INIT="${SWT_INIT:-haar}"
SWT_LEVELS="${SWT_LEVELS:-2}"
LPF="${LPF:-15}"
FREQ_TOP_KS="${FREQ_TOP_KS:-8 16}"
ATTN_TOP_KS="${ATTN_TOP_KS:-3 5}"
ALPHAS="${ALPHAS:-0.5 0.7}"
REVIN="${REVIN:-1}"
AFFINE="${AFFINE:-0}"
SUBTRACT_LAST="${SUBTRACT_LAST:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-./.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] Python interpreter not found: ${PYTHON}"
  echo "Please create venv first: python3 -m venv .venv"
  exit 1
fi

"${PYTHON}" - <<'PY'
import sys
try:
    import torch
    print(f"[ENV] python={sys.executable}")
    print(f"[ENV] torch={torch.__version__}")
except Exception as e:
    print(f"[ERROR] Failed to import torch: {e}")
    raise
PY

for SEQ_LEN in 96 360 720; do
  for PRED_LEN in 96 192; do
    for FREQ_TOP_K in ${FREQ_TOP_KS}; do
      for ATTN_TOP_K in ${ATTN_TOP_KS}; do
        for ALPHA in ${ALPHAS}; do
          MODEL_ID="${DATA}_${MODEL}_sl${SEQ_LEN}_pl${PRED_LEN}_${SWT_INIT}_l${SWT_LEVELS}_k${LPF}_f${FREQ_TOP_K}_a${ATTN_TOP_K}_alpha${ALPHA}_r${REVIN}_af${AFFINE}_slast${SUBTRACT_LAST}"
          echo "======================"
          echo "SEQ_LEN=${SEQ_LEN} PRED_LEN=${PRED_LEN} FREQ_TOP_K=${FREQ_TOP_K} ATTN_TOP_K=${ATTN_TOP_K} ALPHA=${ALPHA} REVIN=${REVIN} AFFINE=${AFFINE} SUBTRACT_LAST=${SUBTRACT_LAST}"
          echo "MODEL_ID=${MODEL_ID}"
          echo "SWT_INIT=${SWT_INIT} SWT_LEVELS=${SWT_LEVELS} LPF=${LPF}"
          echo "======================"

          "${PYTHON}" run_longExp.py \
            --is_training 1 \
            --model ${MODEL} \
            --data ${DATA} \
            --root_path ${ROOT_PATH} \
            --data_path ${DATA_PATH} \
            --features M \
            --target OT \
            --seq_len ${SEQ_LEN} \
            --label_len 48 \
            --pred_len ${PRED_LEN} \
            --embed learned \
            --freq h \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model 512 \
            --e_layers 2 \
            --d_ff 2048 \
            --factor 1 \
            --dropout 0.05 \
            --lpf ${LPF} \
            --swt_levels ${SWT_LEVELS} \
            --swt_init ${SWT_INIT} \
            --freq_top_k ${FREQ_TOP_K} \
            --attn_top_k ${ATTN_TOP_K} \
            --alpha ${ALPHA} \
            --revin ${REVIN} \
            --affine ${AFFINE} \
            --subtract_last ${SUBTRACT_LAST} \
            --segment_num 24 \
            --des "WaveMix" \
            --model_id ${MODEL_ID} \
            --checkpoints ${CHECKPOINTS} \
            --use_gpu True \
            --gpu ${GPU} \
            --use_multi_gpu 0 \
            --itr 1 \
            --train_epochs ${TRAIN_EPOCHS} \
            --batch_size ${BATCH_SIZE}
        done
      done
    done
  done
done