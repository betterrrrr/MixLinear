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

# Force using GPU 3 only
GPU=3

# Validate GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$GPU" -ge "$GPU_COUNT" ]; then
  echo "ERROR: requested fixed GPU 3 is out of range (available $GPU_COUNT)."
  exit 1
fi

echo "Using GPU $GPU"

model_name=MaxDLinear
#model_name=SparseTSF
#MixLinear

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
alpha=0.5
seq_len=720
lpf=1
swt_init='db2'  # 可选值: random, haar, db2
for lpf in 1 5  
do
for alpha in  0.95
do
for pred_len in 96 192
# 336 720
do
  "$PYTHON_BIN" -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --swt_init $swt_init \
    --gpu $GPU \
    --itr 1 --batch_size 256 --learning_rate 0.03 > logs/${model_name}_${data_name}_${pred_len}_${lpf}_${alpha}_${swt_init}_SWT.log  &
done
done
done
