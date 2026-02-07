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
        
    def forward(self, z, y, mu_raw, gates, kernel=None):
        """
        z: ProximalNet 的输出 (或上一阶段的恢复图)
        y: 原始输入 (有雨图)
        mu_raw: DFA_DUN 中定义的可学习参数
        gates: DFA_DUN 中定义的退化类型预测结果 (概率分布)
        """
        # 1. 首先限制所有输入的范围
        z = torch.clamp(z, -10.0, 10.0)
        y = torch.clamp(y, -10.0, 10.0)
        
        # 2. 安全地计算 mu
        # mu 应该是一个较小的正数，例如 [0.1, 5.0]
        mu = F.softplus(mu_raw) + 1e-3  # 确保 mu > 0，且不会太小导致数值不稳定

        x_add = 0
        x_conv = 0
        x_affine = 0
        
        if gates[:,0].max() > 0.01:  # Identity mode
            # 针对去雨任务的标准 DC 公式
            # 背后逻辑是最小化 ||x - z||^2 + mu * ||x - y||^2
            denom = 1.0 + mu
            numerator = y + mu * z
            x_add = numerator / denom
            
        if gates[:,1].max() > 0.01:  # Convolutional mode
            if kernel is None:
                raise ValueError("Convolutional mode requires a kernel.")
            x_conv = self._fft_solver(y, z, mu, kernel)

        if gates[:,2].max() > 0.01:  # Affine mode
            # 简化版：这里可以根据具体去雾物理模型扩展
            # 目前先用 identity 逻辑替代，但保留分支权重学习
            x_affine = (y + mu * z) / (1.0 + mu) 

        x = x_add * gates[:,0:1].view(-1, 1, 1, 1) + \
            x_conv * gates[:,1:2].view(-1, 1, 1, 1) + \
            x_affine * gates[:,2:3].view(-1, 1, 1, 1)

        return x

    def _fft_solver(self, y, z, mu, kernel):
        # 这里是一个更稳定的频域求解器实现
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
