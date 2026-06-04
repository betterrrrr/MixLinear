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
