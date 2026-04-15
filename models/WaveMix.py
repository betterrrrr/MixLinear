# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
#
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x
#
#
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
#
# class Model(nn.Module):
#     """
#     Decomposition-Linear
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#
#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = configs.individual
#         self.channels = configs.enc_in
#
#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
#                 self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
#
#                 # Use this two lines if you want to visualize the weights
#                 # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#                 # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
#             self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
#
#             # Use this two lines if you want to visualize the weights
#             # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#             # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
#         if self.individual:
#             seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
#             trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
#             for i in range(self.channels):
#                 seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
#                 trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)
#
#         x = seasonal_output + trend_output
#         return x.permute(0,2,1) # to [Batch, Output length, Channel]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SWTDecomp(nn.Module):
    """
    SWT decomposition block used to replace moving-average smoothing.
    """

    def __init__(self, kernel_size, levels=2, init_type='haar', enc_in=1):
        super(SWTDecomp, self).__init__()
        self.kernel_size = max(3, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.levels = max(1, int(levels))
        self.enc_in = enc_in

        self.swt_init = init_type.lower()
        low_coef, high_coef = self._swt_init_filters(self.kernel_size, self.swt_init)
        self.low_filter = nn.Parameter(low_coef)
        self.high_filter = nn.Parameter(high_coef)
        self.detail_weights = nn.Parameter(torch.ones(self.levels))

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
            low = torch.randn(kernel_size) * 0.02
            high = torch.randn(kernel_size) * 0.02

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

    def _normalize_filter(self, filt, zero_mean=False, eps=1e-6):
        if zero_mean:
            filt = filt - filt.mean()
        norm = torch.norm(filt, p=2)
        return filt / (norm + eps)

    def _depthwise_same_conv(self, x, filt, dilation):
        pad = ((self.kernel_size - 1) * dilation) // 2
        x_pad = F.pad(x, (pad, pad), mode='circular')
        weight = filt.view(1, 1, -1).repeat(self.enc_in, 1, 1)
        return F.conv1d(x_pad, weight, groups=self.enc_in, dilation=dilation)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        current = x.permute(0, 2, 1)
        details = []

        low_filt = self._normalize_filter(self.low_filter, zero_mean=False)
        high_filt = self._normalize_filter(self.high_filter, zero_mean=True)

        for level in range(self.levels):
            dilation = 2 ** level
            low = self._depthwise_same_conv(current, low_filt, dilation)
            high = self._depthwise_same_conv(current, high_filt, dilation)
            details.append(high)
            current = low

        detail_stack = torch.stack(details, dim=-1)
        weights = F.softmax(self.detail_weights, dim=0).view(1, 1, 1, -1)
        detail = (detail_stack * weights).sum(dim=-1)
        trend = current
        return detail.permute(0, 2, 1), trend.permute(0, 2, 1)


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = getattr(configs, 'lpf', 25)
        self.decompsition = SWTDecomp(
            kernel_size=kernel_size,
            levels=getattr(configs, 'swt_levels', 2),
            init_type=getattr(configs, 'swt_init', 'db2'),
            enc_in=configs.enc_in,
        )
        self.individual = configs.individual
        self.enc_in = configs.enc_in
        self.period_len = 24

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # self.Linear_Seasonal = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        # self.Linear_Trend = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        # seasonal_init = seasonal_init.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # seasonal_output = self.Linear_Seasonal(seasonal_init)  # bc,w,m
        # seasonal_output = seasonal_output.permute(0, 2, 1).reshape(x.size(0), self.enc_in, self.pred_len)
        #
        # trend_init = trend_init.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # trend_output = self.Linear_Trend(trend_init)  # bc,w,m
        # trend_output = trend_output.permute(0, 2, 1).reshape(x.size(0), self.enc_in, self.pred_len)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

