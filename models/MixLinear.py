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

        # Use shallow SWT levels by default to keep useful local details.
        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))

        # Segment-based low-frequency projection settings.
        segment_num_cfg = int(getattr(configs, 'segment_num', max(1, self.seq_len // max(1, self.period_len))))
        self.segment_num = self._resolve_segment_num(self.seq_len, segment_num_cfg)
        self.seg_len = self.seq_len // self.segment_num

        # Learnable weights for multi-scale detail aggregation.
        self.detail_weights = nn.Parameter(torch.ones(self.swt_levels))

        # Wavelet-based initialization option: random / haar / db2
        self.swt_init = getattr(configs, 'swt_init', 'random').lower()
        low_coef, high_coef = self._swt_init_filters(self.swt_kernel_size, self.swt_init)
        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)

        # Segment-based linear projections for low-frequency trend.
        self.intra_linear = nn.Linear(self.seg_len, self.seg_len, bias=False)
        self.inter_linear = nn.Linear(self.segment_num, self.segment_num, bias=False)

        # Lightweight detail refinement branch.
        self.detail_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)

        # Two prediction heads for trend/detail.
        self.trend_head = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.detail_head = nn.Linear(self.seq_len, self.pred_len, bias=False)

        # Learnable global fusion gate initialized from original alpha.
        alpha_init = min(max(float(self.alpha), 1e-4), 1.0 - 1e-4)
        self.mix_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32))

    def _resolve_segment_num(self, seq_len, requested_segments):
        """Pick a segment count that exactly divides seq_len for stable reshape."""
        seg = max(1, min(int(requested_segments), int(seq_len)))
        while seq_len % seg != 0 and seg > 1:
            seg -= 1
        return seg

    def _swt_init_filters(self, kernel_size, init_type='random'):
        """Initialize SWT analysis filters using wavelet coefficients or random noise."""
        init_type = init_type.lower()
        if init_type == 'haar':
            low = torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], dtype=torch.float32)
            high = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], dtype=torch.float32)
        elif init_type in ('db2', 'daubechies'):
            low = torch.tensor([0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512603], dtype=torch.float32)
            high = torch.tensor([-0.1294095225512603, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341], dtype=torch.float32)
        else:
            return torch.randn(kernel_size) * 0.02, torch.randn(kernel_size) * 0.02

        if kernel_size == low.numel():
            return low.clone(), high.clone()
        elif kernel_size < low.numel():
            return low[:kernel_size].clone(), high[:kernel_size].clone()

        pad = kernel_size - low.numel()
        left = pad // 2
        right = pad - left
        low_padded = F.pad(low, (left, right), mode='constant', value=0.0)
        high_padded = F.pad(high, (left, right), mode='constant', value=0.0)
        return low_padded, high_padded

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

        # Weighted aggregation preserves multi-resolution information better than uniform averaging.
        detail_stack = torch.stack(details, dim=-1)  # [B, C, S, L]
        weights = F.softmax(self.detail_weights, dim=0).view(1, 1, 1, -1)
        detail = (detail_stack * weights).sum(dim=-1)
        trend = current
        return trend, detail

    def forward(self, x):
        # Normalization: b,s,c -> mean-centered b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        trend, detail = self._learnable_swt(x)

        # Trend branch: segment-wise factorized linear projection.
        bsz, channels, seq_len = trend.shape
        trend_seg = trend.reshape(bsz, channels, self.segment_num, self.seg_len)
        trend_intra = self.intra_linear(trend_seg)
        trend_inter = self.inter_linear(trend_intra.transpose(-1, -2)).transpose(-1, -2)
        trend_feat = trend_inter.reshape(bsz, channels, seq_len)

        # Detail branch: cheap nonlinear refinement for sparse local fluctuations.
        detail_feat = self.detail_linear(F.gelu(detail))

        # Predict each branch in time domain: [B, C, S] -> [B, C, P] -> [B, P, C]
        trend_pred = self.trend_head(trend_feat).permute(0, 2, 1)
        detail_pred = self.detail_head(detail_feat).permute(0, 2, 1)

        mix_alpha = torch.sigmoid(self.mix_logit)
        x_out = mix_alpha * trend_pred + (1.0 - mix_alpha) * detail_pred + seq_mean

        return x_out[:, :self.pred_len, :]





