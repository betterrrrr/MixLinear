import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """MixDLinear 增强版 (Learnable Everything)

    在原有的 DLinear + SWT 架构基础上，将更多刚性物理操作转化为可学习参数：
    1) 可学习的分解核 (原有)
    2) 可学习的多尺度细节融合 (原有)
    3) 可学习的每通道高频去噪阈值 (新增)
    4) 可学习的趋势与细节重构比例 (新增)
    5) 可学习的未来均值漂移补偿 (新增)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.lpf = getattr(configs, 'lpf', 15)
        self.swt_kernel_size = max(3, int(self.lpf))
        if self.swt_kernel_size % 2 == 0:
            self.swt_kernel_size += 1

        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))

        self.swt_init = getattr(configs, 'swt_init', 'haar').lower()
        low_coef, high_coef = self._swt_init_filters(self.swt_kernel_size, self.swt_init)

        # [原有学习点 1] 分解核
        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)
        
        # [原有学习点 2] 多层细节的融合权重
        self.detail_weights = nn.Parameter(torch.ones(self.swt_levels) / self.swt_levels)

        # ---------------------------------------------------------
        # [新增学习点 3] 可学习的高频软阈值去噪 (Learnable Soft Thresholding)
        # 为每个通道(Channel)独立学习一个降噪门槛，初始值设为极小值(0.01)
        self.shrinkage_threshold = nn.Parameter(torch.ones(1, self.enc_in, 1) * 0.01)
        
        # [新增学习点 4] 可学习的重构权重 (Learnable Recombination)
        # 允许模型学习趋势和细节在最终预测结果中的占比 (初始比例为 1:1)
        self.trend_weight = nn.Parameter(torch.ones(1, self.enc_in, 1))
        self.season_weight = nn.Parameter(torch.ones(1, self.enc_in, 1))

        # [新增学习点 5] 可学习的未来均值补偿 (Learnable Mean Shift Bias)
        # 补偿简单均值归一化在长序列外推时带来的基线漂移误差
        self.mean_shift_bias = nn.Parameter(torch.zeros(1, 1, self.enc_in))
        # ---------------------------------------------------------

        self.individual = configs.individual
        self.period_len = 24
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def _swt_init_filters(self, kernel_size, init_type='haar'):
        init_type = init_type.lower()
        if init_type == 'haar':
            low = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
            high = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], dtype=torch.float32)
        elif init_type in ('db2', 'daubechies'):
            low = torch.tensor([0.48296291, 0.83651630, 0.22414386, -0.12940952], dtype=torch.float32)
            high = torch.tensor([-0.12940952, -0.22414386, 0.83651630, -0.48296291], dtype=torch.float32)
        else:
            low = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
            high = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], dtype=torch.float32)

        if kernel_size == low.numel(): return low.clone(), high.clone()
        if kernel_size < low.numel(): return low[:kernel_size].clone(), high[:kernel_size].clone()
        pad = kernel_size - low.numel()
        left = pad // 2
        right = pad - left
        return F.pad(low, (left, right), mode='constant', value=0.0), F.pad(high, (left, right), mode='constant', value=0.0)

    def _normalize_swt_filter(self, filt, zero_mean=False, eps=1e-6):
        if zero_mean:
            filt = filt - filt.mean()
        norm = torch.norm(filt, p=2)
        return filt / (norm + eps)

    def _depthwise_same_conv(self, x, filt, dilation):
        pad = ((self.swt_kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def _simple_swt(self, x):
        current = x
        details = []

        low_filt = self._normalize_swt_filter(self.low_filter, zero_mean=False)
        high_filt = self._normalize_swt_filter(self.high_filter, zero_mean=True)

        for level in range(self.swt_levels):
            dilation = 2 ** level
            low = self._depthwise_same_conv(current, low_filt, dilation)
            high = self._depthwise_same_conv(current, high_filt, dilation)
            details.append(high)
            current = low

        detail_stack = torch.stack(details, dim=-1)  
        detail = (detail_stack * self.detail_weights.view(1, 1, 1, -1)).sum(dim=-1)
        
        # ---------------------------------------------------------
        # [应用点 3] 软阈值去噪：剥离小于阈值的噪音，保留真实规律
        thresh = torch.abs(self.shrinkage_threshold)
        detail = torch.sign(detail) * F.relu(torch.abs(detail) - thresh)
        # ---------------------------------------------------------
        
        trend = current
        return trend, detail

    def forward(self, x):
        # 去均值
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1) # [B, C, L]
        
        # 分解
        trend_init, seasonal_init = self._simple_swt(x)

        # 线性映射
        seasonal_output = self.Linear_Seasonal(seasonal_init) # [B, C, pred_len]
        trend_output = self.Linear_Trend(trend_init)          # [B, C, pred_len]

        # ---------------------------------------------------------
        # [应用点 4] 加权融合：打破 1:1 的死板相加
        x = (seasonal_output * self.season_weight) + (trend_output * self.trend_weight)
        # ---------------------------------------------------------
        
        x = x.permute(0, 2, 1) # [B, pred_len, C]

        # ---------------------------------------------------------
        # [应用点 5] 均值补偿：在加回历史均值的基础上，允许模型自我修正未来均值
        return x + seq_mean + self.mean_shift_bias
        # ---------------------------------------------------------