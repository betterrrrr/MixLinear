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

        self.lpf = getattr(configs, 'lpf', 15)
        self.kernel_size = max(3, int(self.lpf))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))
        self.swt_init = getattr(configs, 'swt_init', 'db2').lower()
        low_coef, high_coef = self._init_filters(self.kernel_size, self.swt_init)

        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)

        r = max(2, min(8, self.enc_in // 32))
        self.use_ch_mix = 21 <= self.enc_in < 300
        if self.use_ch_mix:
            self.ch_down = nn.Linear(self.enc_in, r, bias=False)
            self.ch_up = nn.Linear(r, self.enc_in, bias=False)

        n_paths = self.swt_levels + 1
        self.heads = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len)
            for _ in range(n_paths)
        ])

        self.level_weights = nn.Parameter(torch.ones(n_paths) / n_paths)
        self.attn_alphas = nn.Parameter(torch.full((n_paths,), -3.0))

    def _init_filters(self, kernel_size, init_type):
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

    def _normalize_filters(self, eps=1e-6):
        high = self.high_filter - self.high_filter.mean()
        low_norm = torch.norm(self.low_filter, p=2) + eps
        high_norm = torch.norm(high, p=2) + eps
        return self.low_filter / low_norm, high / high_norm

    def _depthwise_same_conv(self, x, filt, dilation):
        pad = ((self.kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def _decompose(self, x):
        current = x
        levels = []
        low_filt, high_filt = self._normalize_filters()
        for level in range(self.swt_levels):
            dilation = 2 ** level
            low = self._depthwise_same_conv(current, low_filt, dilation)
            high = self._depthwise_same_conv(current, high_filt, dilation)
            levels.append(high)
            current = low
        levels.append(current)
        return levels

    def _channel_attend(self, x, alpha_param):
        xc = x.permute(0, 2, 1).float()
        x_n = F.normalize(xc, dim=-1)
        cos = torch.bmm(x_n, x_n.transpose(1, 2))
        sin = torch.sqrt(F.relu(1.0 - cos ** 2) + 1e-8)
        cos = cos - cos.amax(dim=-1, keepdim=True).detach()
        sin = sin - sin.amax(dim=-1, keepdim=True).detach()
        alpha = torch.sigmoid(alpha_param)
        score = (1 - alpha) * cos + alpha * sin
        attn = F.softmax(score, dim=-1)
        mixed = torch.bmm(attn, xc).permute(0, 2, 1).to(x.dtype)
        return x + mixed

    def forward(self, x):
        B, L, C = x.shape

        seq_mean = x.mean(dim=1, keepdim=True)
        x_norm = (x - seq_mean).permute(0, 2, 1)

        if self.use_ch_mix:
            ch = self.ch_up(self.ch_down(x_norm.permute(0, 2, 1))).permute(0, 2, 1)
            x_norm = x_norm + ch

        levels = self._decompose(x_norm)

        w = torch.softmax(self.level_weights, dim=0)
        y = 0
        for i, comp in enumerate(levels):
            pred = self.heads[i](comp).permute(0, 2, 1)
            pred = self._channel_attend(pred, self.attn_alphas[i])
            y = y + w[i] * pred

        y = y + seq_mean
        return y
