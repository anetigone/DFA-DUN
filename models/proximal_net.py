import torch
import torch.nn as nn
import torch.nn.functional as F

from .dfam import ModulatedConv2d

# ==========================================
# 动态近端算子网络 (ProximalNet / G_theta)
# ==========================================
# models/proximal_net.py
class DynamicResBlock(nn.Module):
    def __init__(self, n_feats, hyper_dim):
        super().__init__()
        self.conv1 = ModulatedConv2d(n_feats, n_feats, 3, hyper_dim=hyper_dim)
        self.conv2 = ModulatedConv2d(n_feats, n_feats, 3, hyper_dim=hyper_dim)
        self.act = nn.LeakyReLU(0.2, inplace=False)  # 避免 inplace
        
    def forward(self, x, hyper_feat):
        # 添加残差连接的缩放因子
        identity = x
        x = self.act(self.conv1(x, hyper_feat))
        x = self.conv2(x, hyper_feat)
        # 使用较小的残差权重
        return identity + 0.1 * x

class ProximalNet(nn.Module):
    def __init__(self, in_nc=3, n_feats=64, hyper_dim=64):
        super().__init__()
        self.head = nn.Conv2d(in_nc, n_feats, 3, 1, 1)
        # 对 head 也使用谱归一化
        self.head = torch.nn.utils.spectral_norm(self.head)
        
        self.body = nn.ModuleList([
            DynamicResBlock(n_feats, hyper_dim) for _ in range(6)
        ])
        
        self.tail = nn.Conv2d(n_feats, in_nc, 3, 1, 1)
        self.tail = torch.nn.utils.spectral_norm(self.tail)
        
        # 添加一个输出缩放因子
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, hyper_feat):
        # 限制输入
        x = torch.clamp(x, -5.0, 5.0)
        
        h = self.head(x)
        for block in self.body:
            h = block(h, hyper_feat)
        
        # 缩小的残差连接
        output = self.tail(h) * self.output_scale + x * 0.9
        return torch.clamp(output, -5.0, 5.0)