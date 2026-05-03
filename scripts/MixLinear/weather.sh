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

# Prevent loading incompatible packages from ~/.local/lib/python*
export PYTHONNOUSERSITE=1

# Force training processes to only see physical GPU3.
export CUDA_VISIBLE_DEVICES=3

model_name=MixLinear

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
alpha=0.01

for seq_len in 96 360 720
do
for lpf in 1
do
for alpha in 0.01 0.5 0.99
do
for pred_len in 96 192
do
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
    --period_len 4 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --gpu 0 \
    --itr 1 --batch_size 64 --learning_rate 0.02 > logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${lpf}_${alpha}.log
done
done
done
done
