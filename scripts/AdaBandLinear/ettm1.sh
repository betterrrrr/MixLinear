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
export CUDA_VISIBLE_DEVICES=0

model_name=AdaBandLinear
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
train_epochs=40
patience=10
batch_size=128
learning_rate=0.03
swt_levels=3
lpf=5

for seq_len in 96 360 720; do
    for pred_len in 96 192; do
        echo ">>> AdaBandLinear v14 ETTm1 sl${seq_len}_pl${pred_len} swt=${swt_levels} <<<"
        "$PYTHON_BIN" -u run_longExp.py             --is_training 1             --root_path $root_path_name             --data_path $data_path_name             --model_id ${model_id_name}_${seq_len}_${pred_len}             --model $model_name             --data $data_name             --features M             --seq_len ${seq_len}             --pred_len ${pred_len}             --period_len 24             --enc_in 7             --train_epochs $train_epochs             --patience $patience             --gpu 0             --itr 1             --iter_max 1             --batch_size $batch_size             --learning_rate $learning_rate             --swt_levels ${swt_levels}             --swt_init db2             --lpf $lpf > logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}_swt${swt_levels}.log
    done
done
