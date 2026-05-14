#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

. "$SCRIPT_DIR/common.sh"
setup_iTransformer_env

model_name=iTransformer

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=Traffic
data_name=custom
enc_in=862

for seq_len in 96 360 720
do
for pred_len in 96 192
do
  log_file="logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
  run_with_gpu_retry "$log_file" "$PYTHON_BIN" -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_${pred_len} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --des 'Exp' \
    --itr 1 \
    --revin 1
done
done
