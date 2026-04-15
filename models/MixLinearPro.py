import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Model(nn.Module):
    """
    MixLinearPro: 在原有 MixLinear 双分支分解的基础上，将静态全局 alpha = sigmoid(mix_logit) 
    替换为「动态特征注意力门控（Dynamic Attention Gating）」。
    它不仅根据历史趋势与高频特征在时间维/通道维的不同表现进行逐点(point-wise)融合，
    还能更好地自适应应对不同时间步、不同变量的动态分配权重。
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, 'period_len', 24)

        self.lpf = getattr(configs, 'lpf', 5)
        # 用法已改变为注意力融合，这里的 alpha 仅保留兼容
        self.alpha = getattr(configs, 'alpha', 0.5)

        self.swt_kernel_size = max(3, int(self.lpf))
        if self.swt_kernel_size % 2 == 0:
            self.swt_kernel_size += 1

        max_levels = max(1, int(math.log2(max(self.seq_len, 2))) - 1)
        self.swt_levels = min(max_levels, max(1, int(getattr(configs, 'swt_levels', 2))))

        segment_num_cfg = int(getattr(configs, 'segment_num', max(1, self.seq_len // max(1, self.period_len))))
        self.segment_num = self._resolve_segment_num(self.seq_len, segment_num_cfg)
        self.seg_len = self.seq_len // self.segment_num

        self.detail_weights = nn.Parameter(torch.ones(self.swt_levels))

        self.swt_init = getattr(configs, 'swt_init', 'random').lower()
        low_coef, high_coef = self._swt_init_filters(self.swt_kernel_size, self.swt_init)
        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)

        # Trend and Detail projections
        self.intra_linear = nn.Linear(self.seg_len, self.seg_len, bias=False)
        self.inter_linear = nn.Linear(self.segment_num, self.segment_num, bias=False)
        self.detail_linear = nn.Linear(self.seq_len, self.seq_len, bias=False)
        
        self.trend_head = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.detail_head = nn.Linear(self.seq_len, self.pred_len, bias=False)

        # ========================================================================= #
        # Pro升级点：跨域注意力特征门控 (Cross-domain Attention Gate)                  #
        # 输入形状为 [B, C, 2*pred_len]，输出 [B, C, pred_len]                          #
        # 我们用这种残差瓶颈层(Bottleneck MLP)让网络基于双支路的预期情况动态调整融合权重 #
        # ========================================================================= #
        hidden_dim = max(8, self.pred_len // 2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.pred_len * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pred_len),
            nn.Sigmoid()
        )

    def _resolve_segment_num(self, seq_len, requested_segments):
        seg = max(1, min(int(requested_segments), int(seq_len)))
        while seq_len % seg != 0 and seg > 1:
            seg -= 1
        return seg

    def _swt_init_filters(self, kernel_size, init_type='random'):
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
        pad = ((self.swt_kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def _learnable_swt(self, x):
        current = x
        details = []

        for level in range(self.swt_levels):
            dilation = 2 ** level
            low = self._depthwise_same_conv(current, self.low_filter, dilation)
            high = self._depthwise_same_conv(current, self.high_filter, dilation)
            details.append(high)
            current = low

        detail_stack = torch.stack(details, dim=-1)
        weights = F.softmax(self.detail_weights, dim=0).view(1, 1, 1, -1)
        detail = (detail_stack * weights).sum(dim=-1)
        trend = current
        return trend, detail

    def forward(self, x):
        # Normalization: [B, S, C] -> mean-centered [B, C, S]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        trend, detail = self._learnable_swt(x)

        # ----- Trend branch: segment-wise factorized linear projection -----
        bsz, channels, seq_len = trend.shape
        trend_seg = trend.reshape(bsz, channels, self.segment_num, self.seg_len)
        trend_intra = self.intra_linear(trend_seg)
        trend_inter = self.inter_linear(trend_intra.transpose(-1, -2)).transpose(-1, -2)
        trend_feat = trend_inter.reshape(bsz, channels, seq_len)

        # ----- Detail branch: cheap nonlinear refinement -----
        detail_feat = self.detail_linear(F.gelu(detail))

        # Predict each branch: [B, C, S] -> [B, C, P]
        trend_pred = self.trend_head(trend_feat)
        detail_pred = self.detail_head(detail_feat)

        # ==========================================================
        # Dynamic Attention Gating Fusion
        # ==========================================================
        # Concat along temporal dimension to capture multi-domain context
        # cat_pred: [B, C, 2P]
        cat_pred = torch.cat([trend_pred, detail_pred], dim=-1)
        
        # Calculate dynamic time-dependent & channel-dependent fusion weight
        # mix_alpha: [B, C, P]
        mix_alpha = self.fusion_gate(cat_pred)
        
        # Apply gate and permute back to [B, P, C] for standard output
        x_out = (mix_alpha * trend_pred + (1.0 - mix_alpha) * detail_pred).permute(0, 2, 1) + seq_mean

        return x_out[:, :self.pred_len, :]