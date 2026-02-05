import torch
import torch.nn as nn
import torch.nn.functional as F
from .dfam import ModulatedConv2d

# ==========================================
# 辅助类：用于图像的 LayerNorm
# ==========================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        # 归一化后应用可学习的 weight 和 bias
        res = (x - mean) / (std + self.eps)
        res = res * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return res

# ==========================================
# 动态近端算子网络 (ProximalNet)
# ==========================================

class DynamicResBlock(nn.Module):
    def __init__(self, n_feats, hyper_dim):
        super().__init__()
        self.norm1 = LayerNorm2d(n_feats)
        self.conv1 = ModulatedConv2d(n_feats, n_feats, 3, hyper_dim=hyper_dim)
        
        self.norm2 = LayerNorm2d(n_feats)
        self.conv2 = ModulatedConv2d(n_feats, n_feats, 3, hyper_dim=hyper_dim)
        
        self.act = nn.LeakyReLU(0.2, inplace=False)
        
        # --- 改进 1：块内可学习缩放因子 ---
        # 初始化为较小值（如 0.1），有助于深度展开网络的初期稳定
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, hyper_feat):
        identity = x
        
        # Pre-norm 结构通常比 Post-norm 更稳定
        out = self.norm1(x)
        out = self.act(self.conv1(out, hyper_feat))
        
        out = self.norm2(out)
        out = self.conv2(out, hyper_feat)
        
        # 应用可学习缩放
        return identity + out * self.res_scale

class ProximalNet(nn.Module):
    def __init__(self, in_nc=3, n_feats=64, hyper_dim=64, num_blocks=6):
        super().__init__()
        # 初始特征提取
        self.head = nn.Conv2d(in_nc, n_feats, 3, 1, 1)
        
        # 动态残差块列表
        self.body = nn.ModuleList([
            DynamicResBlock(n_feats, hyper_dim) for _ in range(num_blocks)
        ])
        
        # 最后的卷积层将特征转回图像空间
        self.tail = nn.Conv2d(n_feats, in_nc, 3, 1, 1)
        
        # --- 改进 2：全局可学习缩放因子 ---
        # 代替之前的 0.9/0.1 固定配比
        # 初始化为 0.1，意味着初始状态下 ProximalNet 只对图像做微调
        self.global_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 依然保留谱归一化在 tail 层，防止生成的残差过大
        self.tail = torch.nn.utils.spectral_norm(self.tail)
        
    def forward(self, x, hyper_feat):
        """
        x: 来自 DC 层的输入 [B, 3, H, W]
        hyper_feat: 来自 DFAM 的调制参数 [B, hyper_dim, 1, 1]
        """
        # 1. 初始投影
        h = self.head(x)
        
        # 2. 经过多个动态残差块
        for block in self.body:
            h = block(h, hyper_feat)
        
        # 3. 得到残差图并应用全局缩放
        # 这里不再使用 clamp，而是让 LayerNorm 和 Learnable Scale 保证稳定性
        res = self.tail(h)
        
        # 最终输出 = DC 层输入 + 学习到的残差 * 全局缩放因子
        out = x + res * self.global_scale
        
        return out