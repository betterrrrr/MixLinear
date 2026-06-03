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
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
train_epochs=40
patience=10
batch_size=128
learning_rate=0.03

for seq_len in 96 360 720; do
    for pred_len in 96 192; do
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
            --period_len 24 \
            --enc_in 321 \
            --train_epochs $train_epochs \
            --patience $patience \
            --gpu 0 \
            --itr 1 \
            --batch_size $batch_size \
            --learning_rate $learning_rate > logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
    done
done
