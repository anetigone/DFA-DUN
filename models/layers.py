import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

# ==========================================
# 数据一致性层 (Data Consistency Layer)
# ==========================================
class DataConsistencyLayer(nn.Module):
    def forward(self, z, y, mu_raw, mode='identity', kernel=None):
        # 限制 mu 在 [1e-4, 10] 之间，防止过小导致除零，过大导致数值飞走
        mu = F.softplus(mu_raw).clamp(min=1e-4, max=10.0)

        if mode == 'identity' or kernel is None:
            return (y + mu * z) / (1 + mu)

        elif mode == 'convolution':
            B, C, H, W = y.shape
            # 强制 kernel 在 FP32 下计算 FFT，避免混合精度导致的溢出
            kernel_fp32 = kernel.float()
            y_fp32 = y.float()
            z_fp32 = z.float()

            K_fft = torch.fft.rfft2(kernel_fp32, s=(H, W), norm='ortho')
            num = torch.conj(K_fft) * torch.fft.rfft2(y_fp32, norm='ortho') + \
                  mu.view(-1,1,1,1) * torch.fft.rfft2(z_fp32, norm='ortho')
            den = torch.abs(K_fft)**2 + mu.view(-1,1,1,1)
            
            # 使用较宽的 eps
            res = torch.fft.irfft2(num / (den + 1e-6), s=(H, W), norm='ortho')
            return res.type_as(y)