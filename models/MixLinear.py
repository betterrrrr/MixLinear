import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        # Keep original config names for compatibility with existing scripts.
        # Here lpf controls SWT kernel size instead of FFT truncation.
        self.lpf = configs.lpf
        self.alpha = configs.alpha

        # Learnable SWT filter length: force odd to keep same-length convolution.
        self.swt_kernel_size = max(3, int(self.lpf))
        if self.swt_kernel_size % 2 == 0:
            self.swt_kernel_size += 1

        # Choose a small number of levels to avoid over-smoothing on short sequences.
        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(3, max_levels)

        # Two learnable analysis filters (shared over variables, depth-wise applied).
        self.low_filter = nn.Parameter(torch.randn(self.swt_kernel_size) * 0.02)
        self.high_filter = nn.Parameter(torch.randn(self.swt_kernel_size) * 0.02)

        # Two prediction heads for trend/detail.
        self.trend_head = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.detail_head = nn.Linear(self.seq_len, self.pred_len, bias=False)

        # Learnable global fusion gate initialized from original alpha.
        alpha_init = min(max(float(self.alpha), 1e-4), 1.0 - 1e-4)
        self.mix_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32))

    def _depthwise_same_conv(self, x, filt, dilation):
        """Depthwise 1D convolution with circular padding and same output length."""
        # x: [B, C, S], filt: [K]
        pad = ((self.swt_kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def _learnable_swt(self, x):
        """Multi-level learnable SWT decomposition.

        Returns:
            trend: [B, C, S]
            detail: [B, C, S] aggregated detail coefficients
        """
        current = x
        details = []

        for level in range(self.swt_levels):
            dilation = 2 ** level
            low = self._depthwise_same_conv(current, self.low_filter, dilation)
            high = self._depthwise_same_conv(current, self.high_filter, dilation)
            details.append(high)
            current = low

        # Mean aggregation keeps scale stable across different levels.
        detail = torch.stack(details, dim=0).mean(dim=0)
        trend = current
        return trend, detail

    def forward(self, x):
        # Normalization: b,s,c -> mean-centered b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        trend, detail = self._learnable_swt(x)

        # Predict each branch in time domain: [B, C, S] -> [B, C, P] -> [B, P, C]
        trend_pred = self.trend_head(trend).permute(0, 2, 1)
        detail_pred = self.detail_head(detail).permute(0, 2, 1)

        mix_alpha = torch.sigmoid(self.mix_logit)
        x_out = mix_alpha * trend_pred + (1.0 - mix_alpha) * detail_pred + seq_mean

        return x_out[:, :self.pred_len, :]





