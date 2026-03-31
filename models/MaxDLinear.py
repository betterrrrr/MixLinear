import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """MaxDLinear 主体。

    思路与 DLinear 一致，但分解算子换为固定 SWT：
    1) 先做序列分解，得到细节项与趋势项；
    2) 分别使用线性层做从历史窗口到预测窗口的映射；
    3) 两路结果相加得到最终预测。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 输入序列长度与预测长度。
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # 使用 lpf 控制 SWT 的固定滤波器长度（与现有脚本参数兼容）。
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

        # 固定分解核，不参与训练。
        self.register_buffer('low_filter', low_coef)
        self.register_buffer('high_filter', high_coef)

        # individual 在当前实现里未启用，保留是为了兼容外部配置。
        self.individual = configs.individual

        # 每个周期长度，下面两个变量用于分段版本的预留实现。
        self.period_len = 24

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 当前生效实现：直接在时间维做全长度线性映射。
        # 输入 [B, C, seq_len] -> 输出 [B, C, pred_len]
        # self.Linear_Seasonal = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        # self.Linear_Trend = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

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

    def _depthwise_same_conv(self, x, filt, dilation):
        """使用 circular padding 的 depthwise 1D 卷积，保持时间长度不变。"""
        pad = ((self.swt_kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def _simple_swt(self, x):
        """多层固定 SWT 分解。

        参数:
            x: [B, C, S]
        返回:
            trend: [B, C, S]
            detail: [B, C, S]
        """
        current = x
        details = []

        for level in range(self.swt_levels):
            dilation = 2 ** level
            low = self._depthwise_same_conv(current, self.low_filter, dilation)
            high = self._depthwise_same_conv(current, self.high_filter, dilation)
            details.append(high)
            current = low

        detail_stack = torch.stack(details, dim=-1)  # [B, C, S, L]
        # 简单 SWT: 各层细节做等权平均，不引入可学习参数。
        detail = detail_stack.mean(dim=-1)
        trend = current
        return trend, detail

    def forward(self, x):
        # x: [B, L, C]
        # 仅对预测过程做零均值化，减轻绝对幅值与分布漂移对线性映射的影响。
        seq_mean = torch.mean(x, dim=1, keepdim=True)

        # [B, L, C] -> [B, C, L]，分解算子使用固定 SWT。
        x = (x - seq_mean).permute(0, 2, 1)
        trend_init, seasonal_init = self._simple_swt(x)

        # 两个分支独立预测到 pred_len。
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # 以下是历史分段线性版本的代码，当前保留注释以便后续实验切换。
        # seasonal_init = seasonal_init.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # seasonal_output = self.Linear_Seasonal(seasonal_init)  # bc,w,m
        # seasonal_output = seasonal_output.permute(0, 2, 1).reshape(x.size(0), self.enc_in, self.pred_len)
        #
        # trend_init = trend_init.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # trend_output = self.Linear_Trend(trend_init)  # bc,w,m
        # trend_output = trend_output.permute(0, 2, 1).reshape(x.size(0), self.enc_in, self.pred_len)

        # 趋势项 + 季节项重构最终预测，并恢复为 [B, pred_len, C]。
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)

        # 反归一化：将输入均值加回预测结果。
        return x + seq_mean
