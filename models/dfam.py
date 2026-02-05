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

        # 添加层归一化以提高稳定性
        self.ln1 = nn.LayerNorm(in_channels * num_bands)
        self.ln2 = nn.LayerNorm(intermediate_dim)

    def forward(self, x):
        batch, c, h, w = x.shape

        # 层归一化
        x_norm = F.layer_norm(x, x.shape[1:])
        x_fp32 = x_norm.float()
        
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
        # 添加层归一化
        base_feat_vec = self.ln1(base_feat_vec)
        
        # 2. 通过全连接层提取高层特征
        # 这里的 freq_extractor 现在接受 [B, 12] 输入
        base_feat = self.freq_extractor(base_feat_vec) 
        base_feat = self.ln2(base_feat)
        base_feat = torch.tanh(base_feat)
        
        # 3. 得到各个分支输出
        params = self.hyper_params(base_feat).view(batch, -1, 1, 1)
        params = torch.tanh(params) # 限制范围在 [-1, 1]
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
        # 使用谱归一化
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding)
        self.conv = torch.nn.utils.spectral_norm(self.conv)
        
        self.modulation = nn.Linear(hyper_dim, in_nc * 2)
        # 初始化调制层为接近零输出
        nn.init.normal_(self.modulation.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.modulation.bias, 0.0)

    def forward(self, x, hyper_feat):
        # 限制输入范围
        x = torch.clamp(x, -10.0, 10.0)
        
        stats = self.modulation(hyper_feat.view(x.shape[0], -1))
        # 使用 sigmoid 确保 gamma 在 (0,1) 范围内，beta 在 (-0.5,0.5)
        stats = torch.sigmoid(stats) - 0.5
        stats = stats.view(x.shape[0], 2, -1, 1, 1)
        gamma, beta = stats[:, 0] + 1.0, stats[:, 1]  # gamma ~ [0.5, 1.5], beta ~ [-0.5, 0.5]
        
        # 应用调制
        x_mod = x * gamma + beta
        x_mod = torch.clamp(x_mod, -5.0, 5.0)
        
        return self.conv(x_mod)