import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeOperations:
    @staticmethod
    def safe_divide(num, den, eps=1e-6):
        """安全的除法，防止除零和数值溢出"""
        # 限制分母的范围
        den = torch.clamp(den, min=eps, max=1e6)
        return num / den
    
    @staticmethod
    def safe_mul_add(a, b, c, clip_value=10.0):
        """安全的乘加操作"""
        # 先限制 a 和 b 的范围
        a = torch.clamp(a, -clip_value, clip_value)
        b = torch.clamp(b, -clip_value, clip_value)
        return a * b + c
    
    @staticmethod
    def spectral_norm_conv(conv, power_iterations=1):
        """应用谱归一化到卷积层"""
        return torch.nn.utils.spectral_norm(conv, n_power_iterations=power_iterations)