#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

mkdir -p ./logs

if [ -x "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=3

model_name=PatchTST

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
enc_in=7

d_model=128
n_heads=16
d_ff=256
patch_len=16
stride=8
learning_rate=0.0001
lradj=TST

for seq_len in 96 360 720; do
  for pred_len in 96 192; do
    log_file="logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
    "$PYTHON_BIN" -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in $enc_in \
      --d_model $d_model \
      --n_heads $n_heads \
      --d_ff $d_ff \
      --patch_len $patch_len \
      --stride $stride \
      --lradj $lradj \
      --learning_rate $learning_rate \
      --patience 5 \
      --gpu 0 \
      --itr 1 \
      > "$log_file"
  done
done
