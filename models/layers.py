import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

# ==========================================
# 数据一致性层 (Data Consistency Layer)
# ==========================================
# models/layers.py - 完全重写
class DataConsistencyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z, y, mu_raw, mode='identity', kernel=None):
        """
        更安全的实现，防止数值爆炸
        """
        # 1. 首先限制所有输入的范围
        z = torch.clamp(z, -10.0, 10.0)
        y = torch.clamp(y, -10.0, 10.0)
        
        # 2. 安全地计算 mu
        # mu 应该是一个较小的正数，例如 [0.1, 5.0]
        mu = torch.sigmoid(mu_raw) * 4.9 + 0.1  # 限制在 [0.1, 5.0]
        
        if mode == 'identity' or kernel is None:
            # 更稳定的计算公式
            denom = 1.0 + mu
            numerator = y + mu * z
            return numerator / denom
            
        elif mode == 'convolution':
            B, C, H, W = y.shape
            
            # 使用双精度进行FFT计算以提高数值稳定性
            with torch.amp.autocast(device_type='cuda', enabled=False):
                kernel_fp64 = kernel.double()
                y_fp64 = y.double()
                z_fp64 = z.double()
                mu_fp64 = mu.double()
                
                K_fft = torch.fft.rfft2(kernel_fp64, s=(H, W), norm='ortho')
                num = torch.conj(K_fft) * torch.fft.rfft2(y_fp64, norm='ortho') + \
                      mu_fp64.view(-1, 1, 1, 1) * torch.fft.rfft2(z_fp64, norm='ortho')
                den = torch.abs(K_fft)**2 + mu_fp64.view(-1, 1, 1, 1)
                
                # 更安全的除法
                den = torch.clamp(den, min=1e-10, max=1e10)
                result = torch.fft.irfft2(num / den, s=(H, W), norm='ortho')
                
                # 转换回原始精度
                result = result.type_as(y)
            
            # 限制输出范围
            return torch.clamp(result, -5.0, 5.0)