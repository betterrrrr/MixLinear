import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Decomposition-Linear with SWT decomposition.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.enc_in = configs.enc_in
        self.use_revin = bool(getattr(configs, 'revin', 1))
        self.affine = bool(getattr(configs, 'affine', 0))
        self.subtract_last = bool(getattr(configs, 'subtract_last', 0))
        if self.use_revin:
            self.revin_layer = RevIN(self.enc_in, affine=self.affine, subtract_last=self.subtract_last)

        # Reuse existing arguments for SWT settings.
        self.lpf = getattr(configs, 'lpf', 15)
        self.alpha = float(getattr(configs, 'alpha', 0.5))
        self.swt_kernel_size = max(3, int(self.lpf))
        if self.swt_kernel_size % 2 == 0:
            self.swt_kernel_size += 1

        # Track A: MixLinear-style frequency truncation count for trend branch.
        self.freq_top_k = max(1, int(getattr(configs, 'freq_top_k', self.lpf)))

        # Track B: inverted attention with myopic truncation on seasonal branch.
        self.attn_top_k = max(1, int(getattr(configs, 'attn_top_k', max(1, self.enc_in // 2))))
        self.attn_dropout = nn.Dropout(float(getattr(configs, 'dropout', 0.0)))
        self.seasonal_q = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.seasonal_k = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.seasonal_v = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.seasonal_gate_logit = nn.Parameter(torch.full((self.enc_in,), -2.2))

        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))
        self.swt_init = getattr(configs, 'swt_init', 'haar').lower()
        self.detail_weights = nn.Parameter(torch.ones(self.swt_levels) / self.swt_levels)

        low_coef, high_coef = self._swt_init_filters(self.swt_kernel_size, self.swt_init)
        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def _swt_init_filters(self, kernel_size, init_type='haar'):
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
        if zero_mean:
            filt = filt - filt.mean()
        norm = torch.norm(filt, p=2)
        return filt / (norm + eps)

    def _depthwise_same_conv(self, x, filt, dilation):
        pad = ((self.swt_kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def _swt_decompose(self, x):
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
        seasonal = (detail_stack * self.detail_weights.view(1, 1, 1, -1)).sum(dim=-1)
        trend = current
        return seasonal, trend

    def _fft_lowpass_trend(self, trend):
        """Track A: FFT low-pass truncation on trend features."""
        spec = torch.fft.rfft(trend, dim=-1)
        cutoff = min(self.freq_top_k, spec.size(-1))
        low_spec = spec.clone()
        low_spec[..., cutoff:] = 0
        return torch.fft.irfft(low_spec, n=self.seq_len, dim=-1)

    def _simpletm_myopic(self, seasonal):
        """Track B: inverted attention with top-k myopic truncation."""
        q = self.seasonal_q(seasonal)
        k = self.seasonal_k(seasonal)
        v = self.seasonal_v(seasonal)

        scale = 1.0 / math.sqrt(max(1, self.seq_len))
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        top_k = min(self.attn_top_k, scores.size(-1))
        if top_k < scores.size(-1):
            topk_idx = torch.topk(scores, k=top_k, dim=-1).indices
            masked_scores = torch.full_like(scores, float('-inf'))
            masked_scores.scatter_(-1, topk_idx, scores.gather(-1, topk_idx))
            scores = masked_scores

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)

    def forward(self, x, x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        # Only the encoder input x is used for this decomposition-linear model.
        # The extra arguments are accepted for compatibility with the common training loop.
        # x: [Batch, Input length, Channel]
        if self.use_revin:
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)
        seasonal_init, trend_init = self._swt_decompose(x)

        trend_track_a = self._fft_lowpass_trend(trend_init)
        trend_feat = self.alpha * trend_track_a + (1.0 - self.alpha) * trend_init

        seasonal_track_b = self._simpletm_myopic(seasonal_init)
        seasonal_gate = torch.sigmoid(self.seasonal_gate_logit.view(1, -1, 1))
        seasonal_feat = seasonal_gate * seasonal_track_b + (1.0 - seasonal_gate) * seasonal_init

        seasonal_output = self.Linear_Seasonal(seasonal_feat)
        trend_output = self.Linear_Trend(trend_feat)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)

        if self.use_revin:
            x = self.revin_layer(x, 'denorm')

        return x  # to [Batch, Output length, Channel]

