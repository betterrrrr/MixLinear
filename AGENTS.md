# MixLinear — AGENTS.md

## 仓库结构

- **入口**: `run_longExp.py` — 单个脚本，所有配置通过 argparse。
- **模型**: `models/*.py`，每个文件导出 `class Model(nn.Module)`。没有 `__init__.py`；`exp/exp_main.py:28-44` 按文件名导入。
- **Shell 脚本**: `scripts/{模型名}/*.sh` — 每个脚本 `cd` 到项目根目录，优先使用 `./.venv/bin/python`（回退到 `python3`），设置 `PYTHONNOUSERSITE=1`，设置 `CUDA_VISIBLE_DEVICES=3`。日志输出到 `./logs/`。
- **数据**: 从 Autoformer Google Drive 下载，CSV 文件放置在 `./dataset/`（被 `.gitignore`）。
- **检查点**: 保存到 `./checkpoints/`（被 `.gitignore`）。
- **结果**: 追加到 `result.txt`（不在 `.gitignore` 中）。
- **脚本生成**: `generate_scripts.py` 为指定模型生成 shell 脚本，写入 `scripts/`。

## 如何运行

```bash
# 单个实验
sh scripts/MixLinear/etth1.sh

# 运行 MixLinear 所有数据集
bash scripts/MixLinear/run_all.sh

# 直接调用（示例，可见 GPU 3）
CUDA_VISIBLE_DEVICES=3 python -u run_longExp.py --is_training 1 --model MixLinear --data ETTh1 --features M --seq_len 96 --pred_len 96 --period_len 24 --enc_in 7 --alpha 0.5 --lpf 1 --gpu 0 --itr 1 --batch_size 256 --learning_rate 0.03
```

## 关键模型约定

- 模型在 `{'Linear', 'TST', 'SparseTSF', 'MixLinear', 'GFFormer', 'FITS'}` 集合中的只接受 `model(batch_x)` — 没有编码器-解码器或时间戳参数。其他模型接受 `model(batch_x, batch_x_mark, dec_inp, batch_y_mark)`。
- MixLinear 系列常用超参数: `--alpha`（时域/频域混合因子）、`--lpf`（低通滤波器/频域分箱数）、`--period_len`（默认 24）、`--swt_init`（random/haar/db2）、`--swt_levels`、`--segment_num`、`--freq_top_k`。
- 种子：固定范围 2023-2032（共 10 个种子）。`--itr` 控制使用多少个种子运行实验。
- 优化器：Adam。学习率调度器：`OneCycleLR` 或自定义 `type3/lradj`。默认学习率因脚本而异（0.02–0.03）。

## 数据细节

- `Dataset_ETT_hour` 划分：训练=12个月，验证=4个月，测试=4个月（每小时一条记录）。
- `Dataset_ETT_minute` 相同但记录数×4（15分钟间隔）。
- `Dataset_Custom` 划分：70/10/20 百分比。
- 特征模式：`M`（多变量→多变量）、`S`（单变量）、`MS`（多变量→单变量）。
- 数据使用 `StandardScaler` 缩放，仅基于训练集拟合。

## 工具

- `summarize.py` 解析 `result.txt`，按（数据集、模型、seq_len、pred_len）输出最佳 MSE/MAE 的 Markdown 表格。
- `generate_scripts.py` 为指定模型跨标准数据集生成 shell 脚本。编辑模板和数据集列表后运行即可。

## 仓库注意事项

- 无测试、无 lint、无类型检查、无 CI。
- 模型注册表在 `exp/exp_main.py` 中 — 添加新模型需要创建 `models/NewModel.py` 并在 `model_dict` 中注册。
- `exp_basic.py:24-31` — GPU 设备逻辑：当外部已设置 `CUDA_VISIBLE_DEVICES`（例如通过 shell 脚本）时，强制使用 `cuda:0` 作为逻辑设备。
- `FITS` 在注册表中映射到 `DLinear.Model`（不是自己的文件），这很可能是一个 bug/占位。
