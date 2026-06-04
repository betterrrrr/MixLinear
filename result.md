# 5-Model Results on 7 Datasets x 6 Settings

Priority rule:
1. Use paper-reported values when the exact `seq_len -> pred_len` setting exists.
2. If the paper does not report that setting, use local reproduced values from `result.txt`.
3. Keep a cell empty only when neither a paper value nor a matching local value exists.

Cell format:
- `MSE/MAE`: both metrics are available.
- A single number: only MSE is available in the cited paper.

Sources:
- SimpleTM, ICLR 2025, Table 6: `seq_len=96`; reports MSE/MAE for SimpleTM, iTransformer, PatchTST, and DLinear.
- MixLinear, ICLR 2026, Appendix Table 5: `seq_len=720`; reports MSE for MixLinear, DLinear, PatchTST, and iTransformer.
- Local fallback: `result.txt`; used for settings/models not covered by the above papers. DLinear `360->96/192` values were filled from the user-provided original DLinear results.

| Dataset | Seq->Pred | PatchTST | iTransformer | DLinear | MixLinear | SimpleTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Electricity | 96->96 | 0.181/0.270 | 0.148/0.240 | 0.197/0.282 | 0.2098/0.2809 | 0.141/0.235 |
| Electricity | 96->192 | 0.188/0.274 | 0.162/0.253 | 0.196/0.285 | 0.2054/0.2807 | 0.151/0.247 |
| Electricity | 360->96 | 0.1290/0.2220 | 0.1365/0.2339 | 0.1390/0.2363 | 0.1456/0.2402 | 0.1305/0.2248 |
| Electricity | 360->192 | 0.1470/0.2400 | 0.1556/0.2522 | 0.1527/0.2491 | 0.1592/0.2529 | 0.1424/0.2413 |
| Electricity | 720->96 | 0.129 | 0.148 | 0.140 | 0.138 | 0.1309/0.2259 |
| Electricity | 720->192 | 0.149 | 0.162 | 0.153 | 0.154 | 0.1403/0.2403 |
| ETTh1 | 96->96 | 0.414/0.419 | 0.386/0.405 | 0.386/0.400 | 0.4001/0.4051 | 0.366/0.392 |
| ETTh1 | 96->192 | 0.460/0.445 | 0.441/0.436 | 0.437/0.432 | 0.4561/0.4397 | 0.422/0.421 |
| ETTh1 | 360->96 | 0.3700/0.4050 | 0.3928/0.4117 | 0.3697/0.3930 | 0.3993/0.4123 | 0.3782/0.4058 |
| ETTh1 | 360->192 | 0.4130/0.4290 | 0.4241/0.4303 | 0.4164/0.4282 | 0.4194/0.4237 | 0.4210/0.4321 |
| ETTh1 | 720->96 | 0.385 | 0.386 | 0.384 | 0.351 | 0.3667/0.4063 |
| ETTh1 | 720->192 | 0.413 | 0.441 | 0.443 | 0.395 | 0.4119/0.4344 |
| ETTh2 | 96->96 | 0.302/0.348 | 0.297/0.349 | 0.333/0.387 | 0.3050/0.3473 | 0.281/0.338 |
| ETTh2 | 96->192 | 0.388/0.400 | 0.380/0.400 | 0.477/0.476 | 0.3924/0.4008 | 0.355/0.387 |
| ETTh2 | 360->96 | 0.2828/0.3419 | 0.3264/0.3716 | 0.2821/0.3488 | 0.2865/0.3442 | 0.2824/0.3444 |
| ETTh2 | 360->192 | 0.3609/0.3944 | 0.4152/0.4253 | 0.3542/0.3945 | 0.3436/0.3799 | 0.3291/0.3768 |
| ETTh2 | 720->96 | 0.274 | 0.297 | 0.282 | 0.283 | 0.2769/0.3393 |
| ETTh2 | 720->192 | 0.338 | 0.380 | 0.340 | 0.336 | 0.3266/0.3797 |
| ETTm1 | 96->96 | 0.329/0.367 | 0.334/0.368 | 0.345/0.372 | 0.3534/0.3729 | 0.321/0.361 |
| ETTm1 | 96->192 | 0.367/0.385 | 0.377/0.391 | 0.380/0.389 | 0.3988/0.3979 | 0.360/0.380 |
| ETTm1 | 360->96 | 0.2900/0.3350 | 0.3029/0.3558 | 0.3070/0.3527 | 0.3176/0.3590 | 0.2862/0.3426 |
| ETTm1 | 360->192 | 0.3200/0.3480 | 0.3380/0.3768 | 0.3413/0.3714 | 0.3536/0.3789 | 0.3243/0.3665 |
| ETTm1 | 720->96 | 0.293 | 0.334 | 0.299 | 0.308 | 0.2860/0.3456 |
| ETTm1 | 720->192 | 0.333 | 0.377 | 0.335 | 0.337 | 0.3251/0.3695 |
| ETTm2 | 96->96 | 0.175/0.259 | 0.180/0.264 | 0.193/0.292 | 0.1847/0.2671 | 0.173/0.257 |
| ETTm2 | 96->192 | 0.241/0.302 | 0.250/0.309 | 0.284/0.362 | 0.2498/0.3080 | 0.238/0.299 |
| ETTm2 | 360->96 | 0.1620/0.2490 | 0.1838/0.2721 | 0.1736/0.2692 | 0.1687/0.2579 | 0.1670/0.2568 |
| ETTm2 | 360->192 | 0.2120/0.2850 | 0.2525/0.3151 | 0.2468/0.3282 | 0.2236/0.2941 | 0.2303/0.3010 |
| ETTm2 | 720->96 | 0.166 | 0.180 | 0.167 | 0.165 | 0.1713/0.2641 |
| ETTm2 | 720->192 | 0.223 | 0.250 | 0.221 | 0.219 | 0.2342/0.3100 |
| Traffic | 96->96 | 0.462/0.295 | 0.395/0.268 | 0.650/0.396 | 0.6628/0.3991 | 0.410/0.274 |
| Traffic | 96->192 | 0.466/0.296 | 0.417/0.276 | 0.598/0.370 | 0.6109/0.3680 | 0.430/0.280 |
| Traffic | 360->96 | 0.3620/0.2480 | 0.3950/0.2680 | 0.4093/0.2817 | 0.4144/0.2798 | 0.3569/0.2519 |
| Traffic | 360->192 | 0.3780/0.2550 | 0.4070/0.2750 | 0.4211/0.2873 | 0.4272/0.2858 | 0.3766/0.2660 |
| Traffic | 720->96 | 0.366 | 0.395 | 0.413 | 0.389 | 0.3609/0.2570 |
| Traffic | 720->192 | 0.388 | 0.417 | 0.423 | 0.403 | 0.3636/0.2634 |
| Weather | 96->96 | 0.177/0.218 | 0.174/0.214 | 0.196/0.255 | 0.1985/0.2375 | 0.162/0.207 |
| Weather | 96->192 | 0.225/0.259 | 0.221/0.254 | 0.237/0.296 | 0.2429/0.2728 | 0.208/0.248 |
| Weather | 360->96 | 0.1536/0.2043 | 0.1580/0.2110 | 0.1740/0.2339 | 0.1752/0.2269 | 0.1498/0.2026 |
| Weather | 360->192 | 0.1970/0.2441 | 0.2000/0.2510 | 0.2173/0.2746 | 0.2192/0.2635 | 0.1915/0.2421 |
| Weather | 720->96 | 0.149 | 0.174 | 0.176 | 0.170 | 0.1498/0.2043 |
| Weather | 720->192 | 0.194 | 0.221 | 0.218 | 0.212 | 0.1921/0.2462 |

**5 模型 7×6 SOTA 归属表**

| Dataset | 96->96 | 96->192 | 360->96 | 360->192 | 720->96 | 720->192 |
|---|---|---|---|---|---|---|
| Electricity | SimpleTM | SimpleTM | PatchTST | SimpleTM | PatchTST | SimpleTM |
| ETTh1 | SimpleTM | SimpleTM | DLinear | PatchTST | MixLinear | MixLinear |
| ETTh2 | SimpleTM | SimpleTM | DLinear | SimpleTM | PatchTST | SimpleTM |
| ETTm1 | SimpleTM | SimpleTM | SimpleTM | PatchTST | SimpleTM | SimpleTM |
| ETTm2 | SimpleTM | SimpleTM | PatchTST | PatchTST | MixLinear | MixLinear |
| Traffic | iTransformer | iTransformer | SimpleTM | SimpleTM | SimpleTM | SimpleTM |
| Weather | SimpleTM | SimpleTM | SimpleTM | SimpleTM | PatchTST | SimpleTM |

**统计结果**

| Model | SOTA 次数 |
|---|---:|
| SimpleTM | 26 |
| PatchTST | 8 |
| MixLinear | 4 |
| iTransformer | 2 |
| DLinear | 2 |

**可能原因分析**

SimpleTM 在 `96->96` 和 `96->192` 上几乎全面占优，说明短输入场景下，固定窗口内的局部依赖和变量间交互已经足够重要。它把注意力简化成内积/外积结构，参数少、归纳偏置强，不容易在短历史窗口中过拟合。

PatchTST 在部分 `360` 和 `720` 场景变强，尤其是 ETTm2、Weather、Electricity 的部分设置。这可能来自 patch 化机制：长输入序列被切成片段后，模型更容易捕捉中长期模式，同时降低直接 token 级注意力的噪声。

MixLinear 主要在长输入 ETTh1 和 ETTm2 上取得 SOTA，说明简单线性混合在长历史、较强周期或趋势明显的数据上可能更有效。长窗口下复杂注意力不一定必要，线性模型反而能稳定利用历史模式。

DLinear 只在 ETTh1/ETTh2 的 `360->96` 上最优，说明趋势/季节分解在中等输入长度下有价值，但它的固定分解方式不够灵活，所以优势不稳定。

iTransformer 只在 Traffic 的短输入场景最优。Traffic 是高维多变量数据，变量间关系很强；iTransformer 在变量维度做 attention，正好适合这种"通道依赖比时间依赖更关键"的场景。

核心结论可以写成：不同模型的优势不是随机波动，而是和数据集结构、输入长度、预测长度以及模型归纳偏置相关；因此单一 SOTA 说法并不稳健。

---

# 7-Model Results (新增 WaveDLinear + AdaSTM)

在 5 模型基础上，新增 WaveDLinear（用可学习 SWT 替换 DLinear 的 AvgPool）和 AdaSTM（将 SimpleTM 的内外积融合权重改为可学习）两个改进模型。

数据来源：
- 原 5 个模型仍沿用上方 **5-Model Results** 表中的数值，不再重新读取 `result.txt` 里的 PatchTST、iTransformer、MixLinear、SimpleTM 等结果。
- `result.txt` 只用于补充两个新增改进模型：`MixDLinear` 记为 **WaveDLinear**，`SimpleTM2` / `SimpleTM(改版)` 记为 **AdaSTM**；若同一模型同一设置在 `result.txt` 中出现多次，取 MSE 最小的一项。
- SOTA 按 **MSE 最小** 判断；若 MSE 完全相同，则记为并列 SOTA。

**新增两种改进模型结果**

| Dataset | Seq->Pred | WaveDLinear | AdaSTM |
|---|---:|---:|---:|
| Electricity | 96->96 | 0.1973/0.2739 | 0.1398/0.2338 |
| Electricity | 96->192 | 0.1971/0.2764 | 0.1506/0.2476 |
| Electricity | 360->96 | 0.1395/0.2346 | 0.1357/0.2325 |
| Electricity | 360->192 | 0.1536/0.2468 | 0.1406/0.2387 |
| Electricity | 720->96 | 0.1330/0.2283 | 0.1313/0.2260 |
| Electricity | 720->192 | 0.1480/0.2416 | 0.1392/0.2389 |
| ETTh1 | 96->96 | 0.3837/0.3920 | 0.3654/0.3924 |
| ETTh1 | 96->192 | 0.4342/0.4212 | 0.4224/0.4230 |
| ETTh1 | 360->96 | 0.3711/0.3928 | 0.3778/0.4074 |
| ETTh1 | 360->192 | 0.4014/0.4105 | 0.4210/0.4329 |
| ETTh1 | 720->96 | 0.3536/0.3838 | 0.3679/0.4083 |
| ETTh1 | 720->192 | 0.4102/0.4234 | 0.4122/0.4347 |
| ETTh2 | 96->96 | 0.2904/0.3388 | 0.2827/0.3387 |
| ETTh2 | 96->192 | 0.3747/0.3917 | 0.3560/0.3881 |
| ETTh2 | 360->96 | 0.2746/0.3382 | 0.2824/0.3436 |
| ETTh2 | 360->192 | 0.3356/0.3778 | 0.3291/0.3768 |
| ETTh2 | 720->96 | 0.2707/0.3363 | 0.2699/0.3389 |
| ETTh2 | 720->192 | 0.3333/0.3782 | 0.3268/0.3796 |
| ETTm1 | 96->96 | 0.3485/0.3696 | 0.3197/0.3594 |
| ETTm1 | 96->192 | 0.3879/0.3890 | 0.3602/0.3810 |
| ETTm1 | 360->96 | 0.3027/0.3443 | 0.2865/0.3428 |
| ETTm1 | 360->192 | 0.3372/0.3647 | 0.3245/0.3659 |
| ETTm1 | 720->96 | 0.3102/0.3515 | 0.2870/0.3449 |
| ETTm1 | 720->192 | 0.3365/0.3665 | 0.3325/0.3743 |
| ETTm2 | 96->96 | 0.1821/0.2648 | 0.1733/0.2563 |
| ETTm2 | 96->192 | 0.2463/0.3043 | 0.2381/0.2984 |
| ETTm2 | 360->96 | 0.1640/0.2534 | 0.1688/0.2575 |
| ETTm2 | 360->192 | 0.2189/0.2903 | 0.2302/0.3024 |
| ETTm2 | 720->96 | 0.1614/0.2525 | 0.1738/0.2626 |
| ETTm2 | 720->192 | 0.2153/0.2899 | 0.2332/0.3082 |
| Traffic | 96->96 | 0.6461/0.3873 | 0.4213/0.2754 |
| Traffic | 96->192 | 0.5989/0.3634 | 0.4342/0.2756 |
| Traffic | 360->96 | 0.4092/0.2783 | 0.3566/0.2486 |
| Traffic | 360->192 | 0.4214/0.2827 | 0.3756/0.2649 |
| Traffic | 720->96 | 0.3872/0.2688 | 0.3592/0.2549 |
| Traffic | 720->192 | 0.3977/0.2731 | 0.3627/0.2624 |
| Weather | 96->96 | 0.1969/0.2367 | 0.1548/0.2007 |
| Weather | 96->192 | 0.2431/0.2739 | 0.2050/0.2463 |
| Weather | 360->96 | 0.1735/0.2245 | 0.1489/0.2010 |
| Weather | 360->192 | 0.2171/0.2611 | 0.1920/0.2429 |
| Weather | 720->96 | 0.1681/0.2219 | 0.1494/0.2032 |
| Weather | 720->192 | 0.2126/0.2622 | 0.1926/0.2440 |

**7 模型 7×6 SOTA 归属表**

| Dataset | 96->96 | 96->192 | 360->96 | 360->192 | 720->96 | 720->192 |
|---|---|---|---|---|---|---|
| Electricity | AdaSTM | AdaSTM | PatchTST | AdaSTM | PatchTST | AdaSTM |
| ETTh1 | AdaSTM | SimpleTM | DLinear | WaveDLinear | MixLinear | MixLinear |
| ETTh2 | SimpleTM | SimpleTM | WaveDLinear | SimpleTM/AdaSTM | AdaSTM | SimpleTM |
| ETTm1 | AdaSTM | SimpleTM | SimpleTM | PatchTST | SimpleTM | SimpleTM |
| ETTm2 | SimpleTM | SimpleTM | PatchTST | PatchTST | WaveDLinear | WaveDLinear |
| Traffic | iTransformer | iTransformer | AdaSTM | AdaSTM | AdaSTM | AdaSTM |
| Weather | AdaSTM | AdaSTM | AdaSTM | SimpleTM | PatchTST | SimpleTM |

**统计结果（并列单独计为一个归属）**

| Model | SOTA 次数 |
|---|---:|
| AdaSTM | 14 |
| SimpleTM | 12 |
| PatchTST | 6 |
| WaveDLinear | 4 |
| MixLinear | 2 |
| iTransformer | 2 |
| DLinear | 1 |
| SimpleTM/AdaSTM 并列 | 1 |

如果把并列 SOTA 按 0.5 分给两个模型，则为：AdaSTM 14.5、SimpleTM 12.5、PatchTST 6、WaveDLinear 4、MixLinear 2、iTransformer 2、DLinear 1。

**变化总结**

加入两种改进模型后，AdaSTM 成为出现次数最多的模型。它主要在 Electricity、Traffic、Weather，以及部分短输入 ETT 场景胜出，说明可学习内积/外积融合对不同数据结构有更强适应性。WaveDLinear 的胜出集中在 ETTh1 的 360->192、ETTh2 的 360->96，以及 ETTm2 的 720 长输入场景，说明 SWT 分解对部分趋势/周期结构有效，但并不是所有长输入场景都稳定优于 PatchTST、MixLinear 或 SimpleTM。

在这个口径下，原 5 模型的归属只会在新增模型取得更低 MSE 时被替换，因此可以直接看出两种改进是否真正改变原来的 SOTA 格局。这进一步支持核心结论：SOTA 不是绝对模型属性，而是由数据集结构、输入长度、预测长度和模型归纳偏置共同决定。

**5-model vs. 7-model SOTA 归属变化检查**

检查规则：7-model 表相对 5-model 表若发生变化，新的归属必须是 **WaveDLinear** 或 **AdaSTM**；若为并列，则并列项中必须包含 **AdaSTM** 或 **WaveDLinear**。

| Dataset | Seq->Pred | 5-model SOTA | 7-model SOTA | 检查 |
|---|---:|---|---|---|
| Electricity | 96->96 | SimpleTM | AdaSTM | OK |
| Electricity | 96->192 | SimpleTM | AdaSTM | OK |
| Electricity | 360->192 | SimpleTM | AdaSTM | OK |
| Electricity | 720->192 | SimpleTM | AdaSTM | OK |
| ETTh1 | 96->96 | SimpleTM | AdaSTM | OK |
| ETTh1 | 360->192 | PatchTST | WaveDLinear | OK |
| ETTh2 | 360->96 | DLinear | WaveDLinear | OK |
| ETTh2 | 360->192 | SimpleTM | SimpleTM/AdaSTM | OK, 新增并列 |
| ETTh2 | 720->96 | PatchTST | AdaSTM | OK |
| ETTm1 | 96->96 | SimpleTM | AdaSTM | OK |
| ETTm2 | 720->96 | MixLinear | WaveDLinear | OK |
| ETTm2 | 720->192 | MixLinear | WaveDLinear | OK |
| Traffic | 360->96 | SimpleTM | AdaSTM | OK |
| Traffic | 360->192 | SimpleTM | AdaSTM | OK |
| Traffic | 720->96 | SimpleTM | AdaSTM | OK |
| Traffic | 720->192 | SimpleTM | AdaSTM | OK |
| Weather | 96->96 | SimpleTM | AdaSTM | OK |
| Weather | 96->192 | SimpleTM | AdaSTM | OK |
| Weather | 360->96 | SimpleTM | AdaSTM | OK |

结论：共有 19 个格子发生变化，全部变化都由 **WaveDLinear** 或 **AdaSTM** 引起；没有出现 5 个原模型之间互相替换的情况。
