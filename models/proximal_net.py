import torch
import torch.nn as nn
import torch.nn.functional as F

from .dfam import ModulatedConv2d

# ==========================================
# 动态近端算子网络 (ProximalNet / G_theta)
# ==========================================
class DynamicResBlock(nn.Module):
    def __init__(self, n_feats, hyper_dim):
        super().__init__()
        self.conv1 = ModulatedConv2d(n_feats, n_feats, 3, hyper_dim=hyper_dim)
        self.conv2 = ModulatedConv2d(n_feats, n_feats, 3, hyper_dim=hyper_dim)

    def forward(self, x, hyper_feat):
        res = F.relu(self.conv1(x, hyper_feat))
        res = self.conv2(res, hyper_feat)
        return x + res

class ProximalNet(nn.Module):
    """
    一个简化的动态ResNet/U-Net作为近端算子
    """
    def __init__(self, in_nc=3, n_feats=64, hyper_dim=64):
        super().__init__()
        self.head = nn.Conv2d(in_nc, n_feats, 3, 1, 1)
        self.body = nn.ModuleList([
            DynamicResBlock(n_feats, hyper_dim) for _ in range(3)
        ])
        self.tail = nn.Conv2d(n_feats, in_nc, 3, 1, 1)

    def forward(self, x, hyper_feat):
        h = self.head(x)
        for block in self.body:
            h = block(h, hyper_feat)
        return self.tail(h) + x