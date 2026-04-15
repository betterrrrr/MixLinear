#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MixDLinear
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

for seq_len in 96 360 720
do
    for pred_len in 96 192
    do
        # 动态计算滤波参数，防止维度不匹配 (基于此前MixLinear的修复经验)
        period_len=24
        max_lpf=$((seq_len / period_len))
        if [ $max_lpf -gt 15 ]; then
            max_lpf=15
        elif [ $max_lpf -le 0 ]; then
            max_lpf=1
        fi

        CUDA_VISIBLE_DEVICES=3 .venv/bin/python3 -u run_longExp.py \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id ${model_id_name}_${seq_len}_${pred_len} \
            --model $model_name \
            --data $data_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --des 'test' \
            --itr 1 \
            --batch_size 128 \
            --learning_rate 0.03 \
            --train_epochs 40 \
            --patience 10 \
            --lradj type3 \
            --alpha 0.95 \
            --period_len $period_len \
            --lpf $max_lpf \
            --swt_init db2 \
            --swt_levels 3 \
            --use_gpu True \
            --gpu 0 > logs/${model_name}_${data_name}_${seq_len}_${pred_len}.log 
    done
done
