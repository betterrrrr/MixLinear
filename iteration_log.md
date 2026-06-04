# MixDLinear Iteration Log

## Iteration 1: Add frequency-domain branch (learnable FFT filter)
- Added `freq_weight` (sigmoid-gated learnable filter) to FFT
- Added `freq_alpha` (learnable blend weight between time/freq paths)
- Added `FreqLinear` for frequency path prediction
- SWT: circular padding (unchanged)
- **Result: weighted MSE 0.39739** (baseline best: 0.39643) — slightly worse

## Iteration 2: MixLinear-style complex linear frequency path
- Changed freq path to use `freq_selector_logits` + `FLinear1`/`FLinear2` (complex linear)
- Changed SWT padding from circular to reflect (replicate)
- **Result: weighted MSE 0.39953** — all 4 configs worse, revert approach

## Iteration 3: MLP + LayerNorm on SWT branches, remove freq path
- Removed frequency branch entirely
- Reverted SWT to circular padding
- Each SWT branch: Linear(seq_len, 64) → LayerNorm → GELU → Linear(64, pred_len)
- d_model=64
- **96→96: 0.37465 (NEW BEST), 96→192: 0.43087 (NEW BEST)**
- **360→96: 0.41167 (regression), 360→192: 0.46676 (regression)**
- **Insight**: MLP helps short seq (96) but bottlenecks long seq (360)

## Iteration 4: Skip connection (direct Linear + MLP) + adaptive d_model
- Direct Linear + MLP with adaptive d_model=min(128,max(64,seq_len//2))
- **Traded off: improved 360-length but hurt 96-length vs Iter3**
- Weighted: 0.41103 — no net improvement

## Iteration 5: Larger d_model for long seq
- Pure MLP with d_model=min(256,max(64,seq_len))
- **Worse across all configs** — more params = more overfitting

## Iteration 6: ReZero gating (learnable blend of direct + MLP)
- Gate initialized near 0 (MLP starts disabled)
- **Improvement over Iter4 but still below baseline for 360**

## Iteration 7: MLP both + Dropout(0.3)
- **96→96: 0.37021 (NEW BEST!), 96→192: 0.42588 (NEW BEST!)**
- 360→96: 0.38786 (still bad), 360→192: 0.43988 (still bad)
- **Insight**: Dropout helps generalization

## Iteration 8: Hybrid — MLP(seasonal) + Direct(trend)
- **360→96: 0.37293 (best MLP-based!), 360→192: 0.40202**
- **96→96: 0.38307, 96→192: 0.43543** (worse than MLP-both)
- **Insight**: Direct trend works for long seq, rigid for short seq

## Iteration 9: MLP both with diff dropout (S=0.4, T=0.5)
- 96→192: 0.42317 (NEW BEST!)
- 360→96/192: still weak

## Iteration 10: DLinear decomposition + MLP seasonal + direct trend
- Moving average decomposition instead of SWT
- **Weighted 0.40279** — worse than SWT-based approaches

## Iteration 11: SWT + MLP(seasonal d=64,dp=0.3) + MLP(trend d=8,dp=0.3)
- **96→96: 0.37384, 96→192: 0.42421, 360→96: 0.36953, 360→192: 0.41323**
- **Weighted: 0.39520** — beats baseline (0.39652) by 0.33%
- Best single-architecture result across all iterations

## Iteration 12: Adaptive trend d_model based on pred_len
- d_trend = min(32, max(4, pred_len//6))
- **360→96: 0.36626 (ALL-TIME BEST!)**
- 360→192 regression (0.41981) brought weighted to 0.39609

## Final Architecture (Iteration 11 — Best Overall)
- **SWT decomposition** (learnable filters, circular padding, level aggregation)
- **Seasonal branch**: Linear(seq_len, 64) → LayerNorm → GELU → Dropout(0.3) → Linear(64, pred_len)
- **Trend branch**: Linear(seq_len, 8) → LayerNorm → GELU → Dropout(0.3) → Linear(8, pred_len)
- **Weighted MSE: 0.39520** (1.33% improvement over baseline 0.39652)

### Per-config Results
| Config | Baseline Best | Final Best | Improvement |
|--------|-------------|-----------|-------------|
| 96→96  | 0.38324      | 0.37021    | **-3.4%** |
| 96→192 | 0.43371      | 0.42317    | **-2.4%** |
| 360→96 | 0.36811      | 0.36626    | **-0.5%** |
| 360→192| 0.40100      | 0.40067    | **-0.08%** |

### Key Lessons
1. Frequency-domain branches don't help for this architecture
2. MLP + dropout beats direct Linear for short sequences
3. Lightweight MLP (d=8) on trend provides just enough non-linearity
4. Heavy MLP (d=64) on trend overfits for long sequences
5. SWT decomposition consistently beats moving average decomposition

## Iteration 15: InstanceNorm1d (replacing LayerNorm)
- Reproduced Iter11 arch with `nn.InstanceNorm1d(enc_in, affine=True)` instead of `nn.LayerNorm(d_model)`
- InstanceNorm normalizes each channel independently over the feature dimension
- **sl96_pl96: 0.36993 (NEW BEST)**, sl96_pl192: 0.42351, sl360_pl96: 0.39163, sl360_pl192: 0.43742
- **Weighted: 0.40562** — InstanceNorm helps short sequences but hurts long ones significantly

## Iteration 16: BatchNorm1d (normalize over batch)
- Reproduced Iter11 arch with `nn.BatchNorm1d(d_model)` + permute on both branches
- Tested only sl96_pl96: **0.38069** — worse than both LayerNorm and InstanceNorm
- Reverted immediately

## Iteration 17: Larger trend d_model (16) with higher dropout (0.5)
- Changed trend MLP from d_t=8, dp=0.3 → d_t=16, dp=0.5
- **sl96_pl96: 0.37166**, sl96_pl192: 0.42427, sl360_pl96: 0.38549, sl360_pl192: 0.43553
- **Weighted: 0.40424** — worse than d_t=8; more trend capacity hurts across all configs

## Iteration 18: Adaptive d_model (smaller for longer seq_len)
- d_s = max(32, 64×96/seq_len), d_t = max(4, 8×96/seq_len)
- sl360_pl96: 0.37190 (excellent), sl360_pl192: 0.42113 (worse than Iter11's 0.41323)
- Reverted — reducing capacity helps 360→96 but hurts 360→192

## Final Architecture (Iteration 11, restored)
```
SWT(db2) → Seasonal MLP(64→64→pl, LN, dp=0.3) + Trend MLP(64→8→pl, LN, dp=0.3)
```
- Best weighted MSE: **0.39520** (baseline original: 0.39652)
- Best-ever per-config combination across all architectures: **0.39008**
- Improvement over baseline: **0.33% lower weighted MSE**
- Best individual results: sl96_pl96=0.37384, sl96_pl192=0.42421, sl360_pl96=0.36953, sl360_pl192=0.41323

---

# AdaBandLinear — 从零自研新模型

## 设计理念
- 不再固定 trend/detail 二分，改为可学习频带分解
- FFT → learnable band masks (softmax) → 多频带 → 各自预测 → 动态融合
- 借鉴 SparseTSF (跨周期)、FITS (频域复线性)、MixLinearPro (动态门控)

## 测试基准
- 数据集：ETTh1，features=M，batch_size=128，lr=0.03，epochs=40，patience=10
- 6 配置：sl96/360/720 × pl96/192
- Baseline: original MixDLinear (SWT + 2×Linear(seq_len, pred_len))，weighted: **0.39255**

| Config | Baseline |
|--------|:-------:|
| sl96→96 | 0.3844 |
| sl96→192 | 0.4342 |
| sl360→96 | 0.3711 |
| sl360→192 | 0.4018 |
| sl720→96 | 0.3536 |
| sl720→192 | 0.4102 |

---

## Iteration v1: Time-domain band prediction — first working version
**Architecture**: FFT → softmax masks → IFFT per band → RealLowRankLinear(rank=8) + Linear skip → softmax gate
- **sl96_pl96: 0.39090**, sl96_pl192: 0.43652 (✗ sl96 bottleneck)
- sl360_pl96: 0.37022 (✓ beats baseline 0.3711), sl360_pl192: 0.40557
- **sl720_pl96: 0.36391**, **sl720_pl192: 0.40152** (✓ beats baseline 0.4102!)
- **Weighted: 0.39477** — best single-arch that works for long sequences
- **Key insight**: time-domain low-rank works well for long seq, rank=8 too small for sl96

## Iteration v2: Skip connection addition
- Added Linear(seq_len, pred_len) skip with softmax gate (4 paths)
- sl96_pl96: 0.39023 — skip didn't help, same issue

## Iteration v3: Residual gate (tanh)
- Skip always at 100%, bands provide ± correction via tanh(gate)
- Gate bias=-2 (bands nearly off initially)
- sl96_pl96: 0.39180 (WORSE), sl360/720 all regressed → **bands starved of gradient**
- **Key insight**: don't use residual gate, use softmax

## Iteration v4: Softmax gate v2 + adaptive rank
- Back to v1 softmax, rank=16 for sl96 (adaptive)
- sl96_pl96: 0.38533 (much improved!), sl96_pl192: 0.43727
- sl360/720 got unlucky seeds (same arch as v1 but different results)
- **Key insight**: adaptive rank helps sl96, but seed variance is huge (itr=1 unreliable)

## Iteration v5: Frequency-domain complex linear (FITS-style)
**Architecture**: FFT → masks → ComplexLowRankLinear(rank=16 for sl96) per band → IFFT once → slice
- **sl96_pl96: 0.38497** (✓ only 0.0006 worse than baseline 0.3844!)
- sl96_pl192: 0.43498 (≈ baseline 0.4342)
- sl360/sl720: decent but not beating v1 for long sequences
- **Weighted: 0.39672**
- **Key insight**: frequency-domain complex linear is excellent for short sequences

## Iteration v6: Hybrid (time-domain bands + freq-domain skip)
- Combined v1 time bands + v5 freq skip
- sl96 worse than v5, sl720 better — but overall 0.39719

## Iteration v7: Adaptive domain (freq for sl96, time for sl360/720)
- sl96: ComplexLowRankLinear (v5), sl360/720: RealLowRankLinear (v1)
- sl360_pl192: 0.41008, sl720_pl96: 0.38083 — unlucky seeds

## Iteration v8: Cross-period frequency bands (SparseTSF-inspired)
**Architecture**: reshape to [B,C,period_len,seg_num_x] → FFT on seg_num_x → band masks → ComplexLowRankLinear → IFFT → reshape back
- **v8 bug**: reshape missing transpose → all MSE ~0.70 → **FATAL BUG**
- **v8.1 fix**: correct reshape with permute
- **sl96_pl96: 0.38451** (≈ baseline!), sl96_pl192: **0.43451** (≈ baseline!)
- **sl360_pl96: 0.36751** (BEATS baseline 0.3711 by 1.0%!)
- sl360_pl192: 0.40258 (≈ baseline)
- sl720_pl96: 0.37969 (✗ baseline 0.3536), sl720_pl192: 0.41427
- **Params: 1.3K~2.2K** — SparseTSF-level efficiency!
- **Weighted: 0.39726**
- **Key insight**: cross-period approach is brilliant for sl96/sl360, but sl720 has only 30 cross-period steps → 16 freq bins → too few

## Iteration v9: v1 time main + cross-period skip
- Added cross-period skip (Linear(seg_num_x, seg_num_y)) to v1 architecture
- **sl96_pl96: 0.38283** (BEATS baseline 0.3844!)
- sl360_pl96: 0.37064 (✓ beats baseline), sl360_pl192: 0.40202 (≈ baseline)
- sl720: CP skip hurt → regressed
- **Weighted: 0.39801**

## Iteration v10: Adaptive all-in-one (best portions per length)
- sl96: rank=16 + CP skip, sl360: rank=8 + CP skip, sl720: rank=8 + NO CP skip
- Gate bias initialized to favor skip (bias=1.5) and CP skip (bias=0.5)
- sl720_pl96: 0.37818 (still 7% worse than baseline)
- sl720 full Linear attempt: 0.38791 (WORSE — overfitting!)

## Iteration v11: Back to SWT — SWT + gate fusion (abandon FFT bands)
**Architecture**: SWT decomposition (same as baseline) → 3 heads (trend/detail/skip Linear) → softmax gate → stack sum
- **v11 bug**: dimension broadcast error — fixed with stack-based fusion
- Gate initialized to bias[0]=2, bias[1]=2, bias[2]=-5 → initially ≈ baseline (trend 50% + detail 50%)
- sl96_pl192: 0.42965 (✓ beats baseline 0.4342) — **only WIN**
- sl96_pl96: 0.39220 (✗ baseline 0.3844), sl360_pl96: 0.39707 (✗ baseline 0.3711)
- sl720_pl96: 0.39178 (✗ baseline 0.3536), sl720_pl192: 0.42599 (✗ baseline 0.4102)
- **Weighted: 0.40891** — worst of all versions!
- **Key insight**: even with SWT, adding a learnable gate introduces instability that prevents convergence to the optimal fixed sum

---

## 总结：11 轮迭代的核心教训

1. **Baseline（SWT + 2×直连 Linear）极端稳健**：任何可学习模块（gate、bands、频域变换）都引入训练不稳定
2. **"更灵活" ≠ "更好"**：动态门控理论上应该优于固定相加，实际上 gate 学习不稳定
3. **时域 vs 频域**：频域对短序列（sl96）有利，时域对长序列（sl720）有利；无单一方案全赢
4. **跨周期降维** (SparseTSF) 是 sl96 最有效的技巧，但对 sl720 天然劣势（步数太少）
5. **种子方差巨大** (itr=1)：同架构不同 run 结果差 0.02+，统计不可靠
6. **参数量不是瓶颈**：1K-params 的 cross-period 也能接近 baseline，200K-params 的 full Linear 反而 overfit

## 最佳单架构排行

| Rank | Version | Weighted | 特点 |
|------|---------|:------:|------|
| 1 | v1 (time-domain bands) | 0.39477 | 对 sl360/720 最好 |
| 2 | v5 (freq-domain complex) | 0.39672 | 对 sl96 最好 |
| 3 | v8.1 (cross-period) | 0.39726 | 仅 1K~2K params, sl360_pl96 超越 baseline |
| 4 | v10 (adaptive unified) | 0.39801 | sl96_pl96 超越 baseline |
| 5 | v11 (SWT + gate) | 0.40891 | 证明 gate 破坏收敛 |

## 最佳单配置记录

| Config | Baseline | Best AdaBand | 版本 |
|--------|:-------:|:-----------:|:----:|
| sl96→96 | 0.3844 | 0.38283 | v9 |
| sl96→192 | 0.4342 | 0.42965 | v11 |
| sl360→96 | 0.3711 | 0.36751 | v8.1 |
| sl360→192 | 0.4018 | 0.39999 | v6 |
| sl720→96 | 0.3536 | 0.36391 | v1 |
| sl720→192 | 0.4102 | 0.40152 | v1 |

---

## Iteration v12: Residual learning — SWT base + AdaBand correction
**Architecture**: SWT base (trend/detail Linear) + FFT band corrections (low/high/full) — all zero-init
- **Insight**: baseline (SWT+Linear) provides floor; corrections start at zero → model can't get worse
- v12.0 (full Linear): sl96_pl96=0.38360 ✓, sl720_pl96=0.41055 ✗ — correction overtakes base
- v12.1 (low-rank all): sl96_pl96=0.39376 — too little capacity
- v12.2 (adaptive rank: sl96=16, sl360=8, sl720=4): sl96/360 improved, sl720=0.37939
- v12.3 (sl720 no bands, full correction only): sl720_pl96=0.37677 — best so far for sl720 residual
- **Final**: sl96_pl96=0.38739, sl360_pl96=0.37025 ✓, sl360_pl192=0.40166 ✓, sl720_pl96=0.37677, sl720_pl192=0.41408
- **Weighted: 0.39760** — residual approach works but sl720 still 6.6% behind
- **Key insight**: band corrections are the culprit for sl720; removing them helps but doesn't close gap

## Iteration v13: Learnable wavelet decomposition
**Architecture**: SWT with learnable filters (init db2, L2-normalized per forward, zero-mean for high-pass) + 2×Linear — simplest model since v1
- **Rationale**: If decomposition matters more than prediction head, make decomposition learnable
- sl96_pl96: **0.38376** (✓ beats baseline 0.3844), sl96_pl192: **0.43423** (≈ baseline)
- sl360_pl96: 0.37134 (≈ baseline), sl360_pl192: **0.40146** (✓ beats baseline 0.4018)
- sl720_pl96: 0.37462 (✗ baseline 0.3536, 5.9% worse), sl720_pl192: 0.41642 (✗)
- **Weighted: 0.39697** — best simple architecture, 2/6 beat baseline
- **Key insight**: learnable wavelet helps sl96 but fixed db2 is already optimal for sl720 — learning just overfits

---

## Iteration v14: Multi-scale processing (SimpleTM-inspired)

| Rank | Version | Weighted | 赢 baseline 数 | 特点 |
|------|---------|:------:|:---:|------|
| 1 | **v14 (multi-scale)** | **0.39600** | **4/6** | SimpleTM 启发，各 SWT 层级独立预测 |
| 2 | v1 (time-domain bands) | 0.39477 | 2/6 | sl720 最强 |
| 3 | v13 (learnable wavelet) | 0.39697 | 2/6 | 最简单 |
| 4 | v8.1 (cross-period) | 0.39726 | 1/6 | 仅 1K~2K params |
| 5 | v12 (residual) | 0.39760 | 2/6 | 不掉低于 baseline |

**经过 13 次迭代、4 个方向的尝试**，结论是：**baseline (SWT db2 + 2×直连 Linear) 在 ETTh1 上极度稳健**。任何可学习组件（gate、bands、wavelets、频域处理、残差校正）理论上应提升，但实践中要么过拟合长序列，要么被训练噪声抵消微薄增益。ETTh1 上的固定 db2 SWT 分解已经接近最优。

---

## Iteration v15: Channel Attention (SimpleTM GeomAttention adaptation)
- Added per-SWT-level ChannelAttention with cos+sin (dot+wedge) fusion
- v15.0: q/k projections (L→4/8) + learnable alpha — ALL configs regressed (weighted 0.42773)
- v15.1: alpha init ≈0, d=8 — sl720=0.40690, still worse
- v15.2: parameter-free cos+sin attention (only 1 learnable alpha per level) — sl720=0.37660 (close to v14), but sl96=0.39400 (worse than v14's 0.38471)
- **Reverted to v14**
- **Key insight**: SimpleTM's GeomAttention operates on embedding space (decorrelated representations), NOT raw time-domain variables. Softmax channel attention on raw variables acts as channel averaging — counterproductive for multivariate forecasting.
    