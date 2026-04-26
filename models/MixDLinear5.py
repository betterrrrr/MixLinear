import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """MixSWT_Bivariate: 双通道解耦混合 + 宏观趋势振幅调制。"""

    def __init__(self, configs):
        super(Model, self).__init__()
        # 输入序列长度与预测长度。
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # 使用 lpf 控制 SWT 的滤波器长度（与现有脚本参数兼容）。
        self.lpf = getattr(configs, 'lpf', 15)
        self.swt_kernel_size = max(3, int(self.lpf))
        if self.swt_kernel_size % 2 == 0:
            self.swt_kernel_size += 1

        # SWT 分解层数，默认 2 层。
        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))

        # 小波类型：haar / db2。若未指定，默认使用更简单稳定的 haar。
        self.swt_init = getattr(configs, 'swt_init', 'haar').lower()
        low_coef, high_coef = self._swt_init_filters(self.swt_kernel_size, self.swt_init)

        # ---------------------------------------------------------
        # [修改点 1] 将固定分解核改为可学习参数，使其在训练中自动调整
        # 原代码: 
        # self.register_buffer('low_filter', low_coef)
        # self.register_buffer('high_filter', high_coef)
        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)
        
        # [修改点 2] 为不同层级的细节项(detail)引入可学习的融合权重
        self.detail_weights = nn.Parameter(torch.ones(self.swt_levels) / self.swt_levels)
        # ---------------------------------------------------------

        # individual 在当前实现里未启用，保留是为了兼容外部配置。
        self.individual = configs.individual

        # 每个周期长度，下面两个变量用于分段版本的预留实现。
        self.period_len = 24

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 当前生效实现：直接在时间维做全长度线性映射。
        # 输入 [B, C, seq_len] -> 输出 [B, C, pred_len]
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # 每个通道一个可学习阈值，用于细节分支软阈值去噪。
        self.shrinkage_threshold = nn.Parameter(torch.ones(1, self.enc_in, 1) * 0.01)

        # 频带解耦式双变量混合。
        self.trend_channel_matrix = nn.Parameter(torch.eye(self.enc_in) * 3.0)
        self.detail_channel_matrix = nn.Parameter(torch.eye(self.enc_in) * 3.0)

        # 幅度依赖调制门控。
        self.modulation_weight = nn.Parameter(torch.zeros(1, 1, self.enc_in))

    def _swt_init_filters(self, kernel_size, init_type='haar'):
        """初始化固定 SWT 分解滤波器。"""
        init_type = init_type.lower()
        if init_type == 'haar':
            low = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
            high = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], dtype=torch.float32)
        elif init_type in ('db2', 'daubechies'):
            low = torch.tensor([
                0.4829629131445341,
                0.8365163037378079,
                0.2241438680420134,
                -0.1294095225512603,
            ], dtype=torch.float32)
            high = torch.tensor([
                -0.1294095225512603,
                -0.2241438680420134,
                0.8365163037378079,
                -0.4829629131445341,
            ], dtype=torch.float32)
        else:
            low = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
            high = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], dtype=torch.float32)

        if kernel_size == low.numel():
            return low.clone(), high.clone()
        if kernel_size < low.numel():
            return low[:kernel_size].clone(), high[:kernel_size].clone()

        pad = kernel_size - low.numel()
        left = pad // 2
        right = pad - left
        low_padded = F.pad(low, (left, right), mode='constant', value=0.0)
        high_padded = F.pad(high, (left, right), mode='constant', value=0.0)
        return low_padded, high_padded

    def _normalize_swt_filter(self, filt, zero_mean=False, eps=1e-6):
        """逐步归一化可学习小波滤波器，避免层间幅值漂移。

        当 zero_mean=True（主要用于高频滤波器）时，先做零均值化，再归一化能量。
        """
        if zero_mean:
            filt = filt - filt.mean()
        norm = torch.norm(filt, p=2)
        return filt / (norm + eps)

    def _depthwise_same_conv(self, x, filt, dilation):
        """使用 circular padding 的 depthwise 1D 卷积，保持时间长度不变。"""
        pad = ((self.swt_kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        # filt 现在是 nn.Parameter，通过 repeat 扩展到各通道，梯度会自动累加更新 filt
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)
    def _simple_swt(self, x):
        """多层可学习 SWT 分解。"""
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

        detail_stack = torch.stack(details, dim=-1)  # [B, C, S, L]
        
        # ---------------------------------------------------------
        # [修改点 3] 应用可学习的权重来融合多层细节，而不是死板的等权平均
        # 原代码: detail = detail_stack.mean(dim=-1)
        detail = (detail_stack * self.detail_weights.view(1, 1, 1, -1)).sum(dim=-1)

        # 可学习软阈值去噪：抑制幅值较小的高频噪声。
        thresh = torch.abs(self.shrinkage_threshold)
        detail = torch.sign(detail) * F.relu(torch.abs(detail) - thresh)
        # ---------------------------------------------------------
        
        trend = current
        return trend, detail

    def forward(self, x):
        # x: [B, L, C]
        # 仅对预测过程做零均值化，减轻绝对幅值与分布漂移对线性映射的影响。
        seq_mean = torch.mean(x, dim=1, keepdim=True)

        # [B, L, C] -> [B, C, L]
        x = (x - seq_mean).permute(0, 2, 1)
        trend_init, seasonal_init = self._simple_swt(x)

        # 两个分支独立预测到 pred_len。
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # 交叉通道混合与幅度调制。
        trend_output_t = trend_output.permute(0, 2, 1)
        seasonal_output_t = seasonal_output.permute(0, 2, 1)

        W_trend = F.softmax(self.trend_channel_matrix, dim=-1)
        W_detail = F.softmax(self.detail_channel_matrix, dim=-1)

        trend_mixed = torch.matmul(trend_output_t, W_trend)
        seasonal_mixed = torch.matmul(seasonal_output_t, W_detail)

        gate = torch.tanh(self.modulation_weight * trend_mixed)
        x_fused = trend_mixed + seasonal_mixed * (1.0 + gate)

        # 反归一化：将输入均值加回预测结果。
        return x_fused + seq_mean