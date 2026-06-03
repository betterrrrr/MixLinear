#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

export PYTHONNOUSERSITE=1

export CUDA_VISIBLE_DEVICES=3

model_name=DLinear
root_path_name=./dataset/
train_epochs=40
patience=10
batch_size=128
learning_rate=0.03

datasets=(
  "ETTh1:ETTh1.csv:ETTh1:7"
  "ETTh2:ETTh2.csv:ETTh2:7"
  "ETTm1:ETTm1.csv:ETTm1:7"
  "ETTm2:ETTm2.csv:ETTm2:7"
  "Electricity:electricity.csv:custom:321"
  "traffic:traffic.csv:custom:862"
  "weather:weather.csv:custom:21"
)

seq_len=360

for dataset in "${datasets[@]}"; do
  IFS=':' read -r model_id data_path data_name enc_in <<< "$dataset"
  for pred_len in 96 192; do
    "$PYTHON_BIN" -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path \
      --model_id ${model_id}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 24 \
      --enc_in $enc_in \
      --train_epochs $train_epochs \
      --patience $patience \
      --gpu 0 \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate > logs/${model_name}_${model_id}_${seq_len}_${pred_len}.log
  done
done
