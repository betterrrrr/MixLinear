import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = configs.individual
        self.ablation_mode = getattr(configs, 'ablation_mode', 'original')

        self.lpf = getattr(configs, 'lpf', 15)
        self.swt_kernel_size = max(3, int(self.lpf))
        if self.swt_kernel_size % 2 == 0:
            self.swt_kernel_size += 1

        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))

        self.swt_init = getattr(configs, 'swt_init', 'haar').lower()
        low_coef, high_coef = self._init_swt_filters(self.swt_kernel_size, self.swt_init)

        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)
        self.detail_weights = nn.Parameter(torch.ones(self.swt_levels) / self.swt_levels)

        d_s = 48
        d_t = 6

        self.S_Linear1 = nn.Linear(self.seq_len, d_s)
        self.S_norm = nn.LayerNorm(d_s)
        self.S_dropout = nn.Dropout(0.3)
        self.S_Linear2 = nn.Linear(d_s, self.pred_len)

        self.T_Linear1 = nn.Linear(self.seq_len, d_t)
        self.T_norm = nn.LayerNorm(d_t)
        self.T_dropout = nn.Dropout(0.3)
        self.T_Linear2 = nn.Linear(d_t, self.pred_len)

    def _init_swt_filters(self, kernel_size, init_type='haar'):
        init_type = init_type.lower()
        if init_type == 'haar':
            low = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
            high = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], dtype=torch.float32)
        elif init_type in ('db2', 'daubechies'):
            low = torch.tensor([0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512603], dtype=torch.float32)
            high = torch.tensor([-0.1294095225512603, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341], dtype=torch.float32)
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
        return F.pad(low, (left, right), mode='constant', value=0.0), F.pad(high, (left, right), mode='constant', value=0.0)

    def _normalize_swt_filter(self, filt, zero_mean=False, eps=1e-6):
        if zero_mean:
            filt = filt - filt.mean()
        return filt / (torch.norm(filt, p=2) + eps)

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
        return current, detail

    def forward(self, x):
        B, L, C = x.shape
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1)

        trend_init, seasonal_init = self._simple_swt(x)

        s = self.S_Linear1(seasonal_init)
        s = self.S_norm(s)
        s = F.gelu(s)
        s = self.S_dropout(s)
        seasonal_output = self.S_Linear2(s)

        t = self.T_Linear1(trend_init)
        t = self.T_norm(t)
        t = F.gelu(t)
        t = self.T_dropout(t)
        trend_output = self.T_Linear2(t)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1) + seq_mean
        return x
