import torch
import torch.nn as nn
from .dfam import DFAM
from .proximal_net import ProximalNet
from .layers import DataConsistencyLayer

# ==========================================
# DUN主干网络 (DFA_DUN)
# ==========================================
class DFA_DUN(nn.Module):
    def __init__(self, K=8, in_nc=3):
        super(DFA_DUN, self).__init__()
        self.K = K
        self.dc_layer = DataConsistencyLayer()

        self.mus = nn.Parameter(torch.ones(K) * 0.5)
        self.dfams = nn.ModuleList([DFAM(in_nc) for _ in range(K)])
        self.prox_nets = nn.ModuleList([ProximalNet(in_nc) for _ in range(K)])

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重,避免数值不稳定"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)    
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                if 'mu' in m.name:
                    nn.init.constant_(m, 1.0)

    def forward(self, y, mode='identity', kernel=None):
        z = y
        all_probs = []

        for k in range(self.K):
            # 1. 获取动态参数和概率
            dynamic_params, gates, pred_kernel = self.dfams[k](z)
            all_probs.append(gates)

            # 2. 软切换逻辑 (Differentiable Soft-Switching)
            # 获取 Index 1 (Blur/Convolution) 的概率
            # gates: [B, 4], p_blur: [B, 1]
            p_blur = gates[:, 1:2] 
            
            # 将 p_blur 扩展为 [B, 1, 1, 1] 以进行像素级相乘
            p_blur_weight = p_blur.unsqueeze(-1).unsqueeze(-1)

            # 计算两条路径
            x_identity = self.dc_layer(z, y, self.mus[k], mode='identity')
            # 这里使用 DFAM 预测出的 kernel 进行盲去模糊尝试
            x_conv = self.dc_layer(z, y, self.mus[k], mode='convolution', kernel=pred_kernel)

            # 融合：如果 p_blur 很高，则 x 更接近 x_conv
            x = p_blur_weight * x_conv + (1.0 - p_blur_weight) * x_identity

            # 3. 近端映射
            # 注意：这里的 ProximalNet 可以接收 y 作为辅助输入（如果之前改过的话）
            z = self.prox_nets[k](x, dynamic_params)

        return z, all_probs