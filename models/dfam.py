import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================
# 动态频率感知模块 (DFAM)
# ==========================================
class DFAM(nn.Module):
    def __init__(self, in_channels, num_bands=4, intermediate_dim=64, kernel_size=15):
        super().__init__()
        self.num_bands = num_bands
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        
        # --- 修改这里：将 Conv2d 改为 Linear ---
        # 因为在频域提取后，特征已经是 [B, C*num_bands] 的向量了
        self.freq_extractor = nn.Sequential(
            nn.Linear(in_channels * num_bands, intermediate_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 分支 1: 生成近端算子的调制参数
        self.hyper_params = nn.Linear(intermediate_dim, intermediate_dim)
        
        # 分支 2: 退化类型预测
        self.degrad_classifier = nn.Sequential(
            nn.Linear(intermediate_dim, 4), 
            nn.Softmax(dim=1)
        )

        # 分支 3: 模糊核预测
        self.kernel_predictor = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, kernel_size * kernel_size),
        )

    def forward(self, x):
        batch, c, h, w = x.shape
        
        # 1. 频域处理 (强制转为 float32 提高数值稳定性)
        x_fp32 = x.float()
        fft_x = torch.fft.rfft2(x_fp32, norm='ortho')
        mag = torch.abs(fft_x)
        
        # 生成频率半径坐标
        u = torch.fft.rfftfreq(h).to(x.device)
        v = torch.fft.fftfreq(w).to(x.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        radius = torch.sqrt(u_grid**2 + v_grid**2)
        
        max_r = radius.max()
        pooled_features = []
        
        for i in range(self.num_bands):
            # 简单的带通掩码
            lower = max_r * (i / self.num_bands)
            upper = max_r * ((i + 1) / self.num_bands)
            mask = ((radius >= lower) & (radius < upper)).float()
            
            # 提取该频段的统计特征 (均值)，形状 [B, C]
            # mask 为 [H, W/2+1], mag 为 [B, C, H, W/2+1]
            feat = (mag * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3)) / (mask.sum() + 1e-6)
            pooled_features.append(feat)
            
        # 拼接特征: [B, num_bands * C] (例如 16 * 12)
        base_feat_vec = torch.cat(pooled_features, dim=1)
        
        # 2. 通过全连接层提取高层特征
        # 这里的 freq_extractor 现在接受 [B, 12] 输入
        base_feat = self.freq_extractor(base_feat_vec) 
        
        # 3. 得到各个分支输出
        params = self.hyper_params(base_feat).view(batch, -1, 1, 1)
        gates = self.degrad_classifier(base_feat)
        raw_kernel = self.kernel_predictor(base_feat)
        
        # 限制 kernel 范围并归一化
        predicted_k = F.softmax(raw_kernel, dim=1).view(-1, 1, self.kernel_size, self.kernel_size)

        return params, gates, predicted_k.type_as(x)

# ==========================================
# 动态权重调制卷积 (Modulated Convolution)
# ==========================================
class ModulatedConv2d(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride=1, padding=1, hyper_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding)
        # 产生缩放(gamma)和偏移(beta)
        self.modulation = nn.Linear(hyper_dim, in_nc * 2) 

    def forward(self, x, hyper_feat):
        # hyper_feat shape: [B, hyper_dim, 1, 1]
        stats = self.modulation(hyper_feat.view(x.shape[0], -1)).view(x.shape[0], 2, -1, 1, 1)
        gamma, beta = stats[:, 0], stats[:, 1]
        
        # 限制 gamma 的范围，防止数值爆炸
        gamma = torch.tanh(gamma) + 1.0 # 范围控制在 [0, 2]
        
        # 作用于输入特征 (类似于 Feature-wise Linear Modulation, FiLM)
        x = x * gamma + beta
        return self.conv(x)