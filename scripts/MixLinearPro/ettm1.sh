#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ -x "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

export PYTHONNOUSERSITE=1
GPU=3

model_name=MixLinearPro

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

period_len=24
swt_init=db2
swt_levels=3
lpf=5
train_epochs=40
patience=10
batch_size=128
learning_rate=0.03
lradj=type3
alpha=0.5 

echo "Starting $model_name experiments on $data_name..."

for seq_len in 96 360 720; do
    for pred_len in 96 192; do
        log_file="logs/${model_name}_${data_name}_${seq_len}_${pred_len}.log"
        echo "Running seq_len=$seq_len pred_len=$pred_len -> $log_file"

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
            --period_len $period_len \
            --enc_in 7 \
            --train_epochs $train_epochs \
            --patience $patience \
            --lradj $lradj \
            --alpha $alpha \
            --lpf $lpf \
            --swt_init $swt_init \
            --swt_levels $swt_levels \
            --use_gpu True \
            --gpu 0 \
            --itr 1 \
            --batch_size $batch_size \
            --learning_rate $learning_rate > "$log_file"
    done
done
echo "All $data_name experiments finished."
