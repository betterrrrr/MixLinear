#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ -x "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

# Prevent loading incompatible packages from ~/.local/lib/python*
export PYTHONNOUSERSITE=1

# Force using GPU 3 only
GPU=3

# Validate GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$GPU" -ge "$GPU_COUNT" ]; then
  echo "ERROR: requested fixed GPU $GPU is out of range (available $GPU_COUNT)."
  exit 1
fi

echo "Using GPU $GPU"

model_name=MixDLinear6
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

alpha=0.95
lpf=5
swt_init='db2'
swt_levels=2

d_model=64
n_heads=4
d_ff=128
dropout=0.1

train_epochs=40
patience=10
batch_size=128
learning_rate=0.0005
lradj=type3

for seq_len in 96 360 720; do
    for pred_len in 96 192; do
        log_file="logs/${model_name}_${data_name}_${seq_len}_${pred_len}_${lpf}_${alpha}_${swt_init}_SWT.log"
        echo "Run: model=$model_name seq_len=$seq_len pred_len=$pred_len log=$log_file"

        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u run_longExp.py \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id ${model_id_name}_${seq_len}_${pred_len} \
            --model $model_name \
            --data $data_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --period_len 24 \
            --enc_in 7 \
            --train_epochs $train_epochs \
            --patience $patience \
            --lradj $lradj \
            --alpha $alpha \
            --lpf $lpf \
            --swt_init $swt_init \
            --swt_levels $swt_levels \
            --d_model $d_model \
            --n_heads $n_heads \
            --d_ff $d_ff \
            --dropout $dropout \
            --use_gpu True \
            --gpu 0 \
            --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file"
    done
done
