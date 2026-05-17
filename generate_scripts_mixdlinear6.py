import os

datasets = [
    {"script_name": "etth1.sh", "data_path": "ETTh1.csv", "model_id": "ETTh1", "data": "ETTh1", "enc_in": 7},
    {"script_name": "etth2.sh", "data_path": "ETTh2.csv", "model_id": "ETTh2", "data": "ETTh2", "enc_in": 7},
    {"script_name": "ettm1.sh", "data_path": "ETTm1.csv", "model_id": "ETTm1", "data": "ETTm1", "enc_in": 7},
    {"script_name": "ettm2.sh", "data_path": "ETTm2.csv", "model_id": "ETTm2", "data": "ETTm2", "enc_in": 7},
    {"script_name": "electricity.sh", "data_path": "electricity.csv", "model_id": "Electricity", "data": "custom", "enc_in": 321},
    {"script_name": "traffic.sh", "data_path": "traffic.csv", "model_id": "traffic", "data": "custom", "enc_in": 862},
    {"script_name": "weather.sh", "data_path": "weather.csv", "model_id": "weather", "data": "custom", "enc_in": 21}
]

template = """SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

model_name=MixDLinear6
root_path_name=./dataset/
data_path_name={data_path}
model_id_name={model_id}
data_name={data}
alpha=0.95
lpf=5
swt_init='db2'
swt_levels=3
train_epochs=40
patience=10
batch_size=128
learning_rate=0.03

for seq_len in 96 360 720; do
    for pred_len in 96 192; do
        "$PYTHON_BIN" -u run_longExp.py \\
            --is_training 1 \\
            --root_path $root_path_name \\
            --data_path $data_path_name \\
            --model_id ${{model_id_name}}_${{seq_len}}_${{pred_len}} \\
            --model $model_name \\
            --data $data_name \\
            --features M \\
            --seq_len $seq_len \\
            --pred_len $pred_len \\
            --period_len 24 \\
            --enc_in {enc_in} \\
            --train_epochs $train_epochs \\
            --patience $patience \\
            --alpha $alpha \\
            --lpf $lpf \\
            --swt_init $swt_init \\
            --swt_levels $swt_levels \\
            --gpu 0 \\
            --itr 1 \\
            --batch_size $batch_size \\
            --learning_rate $learning_rate > logs/${{model_name}}_${{model_id_name}}_${{seq_len}}_${{pred_len}}_${{lpf}}_${{alpha}}_${{swt_init}}.log
    done
done
"""

output_dir = "scripts/MixDLinear6"
os.makedirs(output_dir, exist_ok=True)

for ds in datasets:
    script_content = template.format(
        data_path=ds["data_path"],
        model_id=ds["model_id"],
        data=ds["data"],
        enc_in=ds["enc_in"]
    )
    file_path = os.path.join(output_dir, ds["script_name"])
    with open(file_path, "w") as f:
        f.write(script_content)
    os.chmod(file_path, 0o755)

print("Generated scripts for MixDLinear6 with swt_levels.")
