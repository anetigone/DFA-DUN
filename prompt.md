损失函数爆炸了，结果全是nan
全部代码如下
models/dfa_dun.py
```python
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
```

models/proximal_net.py
```python
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
```

models/layers.py
```python
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

# ==========================================
# 数据一致性层 (Data Consistency Layer)
# ==========================================
class DataConsistencyLayer(nn.Module):
    def __init__(self):
        super(DataConsistencyLayer, self).__init__()

    def forward(self, z, y, mu_raw, mode='identity', kernel=None):
        # 使用 softplus 避免数值爆炸,并限制最大值
        mu = F.softplus(mu_raw).clamp(max=10.0)

        if mode == 'identity' or kernel is None:
            # y = x + n -> x = (y + mu*z) / (1 + mu)
            # 添加数值稳定性保护
            denominator = (1 + mu)
            return (y + mu * z) / denominator.clamp(min=1e-6)

        elif mode == 'convolution':
            # 非盲去模糊公式
            _, _, H, W = y.shape
            K_fft = torch.fft.rfft2(kernel, s=(H, W), norm='ortho')
            # 闭式解：(K* Y + mu*Z) / (|K|^2 + mu)
            num = torch.conj(K_fft) * torch.fft.rfft2(y, norm='ortho') + mu * torch.fft.rfft2(z, norm='ortho')
            den = torch.abs(K_fft)**2 + mu
            # 添加小常数避免除零
            result = torch.fft.irfft2(num / (den + 1e-8), s=(H, W), norm='ortho')
            return result.clamp(-10, 10)  # 限制输出范围
```

models/dfam.py
```python
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
        
        # 基础频率特征提取
        self.freq_extractor = nn.Sequential(
            nn.Conv2d(in_channels * num_bands, intermediate_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 分支 1: 生成近端算子的调制参数
        self.hyper_params = nn.Linear(intermediate_dim, intermediate_dim)
        
        # 分支 2: 退化类型预测 (Mask/Gates)
        # 输出例如: [p_noise, p_blur, p_rain, p_haze]
        self.degrad_classifier = nn.Sequential(
            nn.Linear(intermediate_dim, 4), 
            nn.Softmax(dim=1)
        )

        # 分支 3: 模糊核预测 (只有当判定为模糊时才生效)
        # 生成一个 kernel_size * kernel_size 的空间域核
        self.kernel_predictor = nn.Sequential(
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, kernel_size * kernel_size),
        )

    def forward(self, x):
        batch, c, h, w = x.shape
        fft_x = torch.fft.rfft2(x, norm='ortho')
        
        # 生成频率半径矩阵 (用于切分低中高频)
        # 这里动态生成简单的带通滤波器
        freq_features = []
        # 获取频率坐标
        u = torch.fft.rfftfreq(h).to(x.device)
        v = torch.fft.fftfreq(w).to(x.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        radius = torch.sqrt(u_grid**2 + v_grid**2)
        
        max_r = radius.max()
        for i in range(self.num_bands):
            # 简单的带通掩码：例如 [0, 0.25], [0.25, 0.5] ...
            lower = max_r * (i / self.num_bands)
            upper = max_r * ((i + 1) / self.num_bands)
            mask = ((radius >= lower) & (radius < upper)).float()
            
            filtered_fft = fft_x * mask.unsqueeze(0).unsqueeze(0)
            filtered_x = torch.fft.irfft2(filtered_fft, s=(h, w), norm='ortho')
            freq_features.append(filtered_x)
            
        # 拼接并送入超网络
        feat = torch.cat(freq_features, dim=1) # [B, C*num_bands, H, W]
        base_feat = self.freq_extractor(feat) # [B, intermediate_dim]
        
        # 2. 得到各个分支输出
        params = self.hyper_params(base_feat).view(x.shape[0], -1, 1, 1)
        gates = self.degrad_classifier(base_feat)
        raw_kernel = self.kernel_predictor(base_feat)
        predicted_k = F.softmax(raw_kernel, dim=1).view(-1, 1, self.kernel_size, self.kernel_size)

        return params, gates, predicted_k

# ==========================================
# 动态权重调制卷积 (Modulated Convolution)
# ==========================================
class ModulatedConv2d(nn.Module):
    """
    通过超网络生成的向量来调制卷积层的权重
    """
    def __init__(self, in_nc, out_nc, kernel_size, stride=1, padding=1, hyper_dim=64):
        super().__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 使用 Xavier 初始化,避免数值过大
        self.weight = nn.Parameter(torch.empty(out_nc, in_nc, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.modulation = nn.Linear(hyper_dim, in_nc) # 将超网络输出映射到输入通道数

    def forward(self, x, hyper_feat):
        # hyper_feat: [B, hyper_dim, 1, 1]
        b, c, h, w = x.shape
        
        # 生成权重缩放因子
        style = self.modulation(hyper_feat.view(b, -1)).view(b, 1, c, 1, 1)
        # 调制权重: W' = W * style
        # 这种方式比直接相加 (theta + delta_theta) 在训练上更稳定且参数量小
        weight = self.weight.unsqueeze(0) * style 
        
        # 将 Batch 维度融入 Group 卷积以实现并行计算
        x = x.view(1, b * c, h, w)
        weight = weight.view(b * self.out_nc, self.in_nc, self.kernel_size, self.kernel_size)
        
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=b)
        return out.view(b, self.out_nc, h, w)
```

training/losses.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTermLoss(nn.Module):
    def __init__(self, lambda_freq=0.1, lambda_cls=0.05):
        super(MultiTermLoss, self).__init__()
        self.lambda_freq = lambda_freq
        self.lambda_cls = lambda_cls
        self.charbonnier = L1_Charbonnier_Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred_img, gt_img, pred_probs, gt_labels):
        """
        pred_img: 恢复的图像 [B, 3, H, W]
        gt_img: 干净的图像 [B, 3, H, W]
        pred_probs: DFAM 输出的分类概率, 列表形式 [stage1_prob, stage2_prob, ...]
        gt_labels: 退化类型标签 [B] (例如: 0为恒等, 1为卷积)
        """
        # 检查输入是否包含 NaN
        if torch.isnan(pred_img).any():
            raise ValueError("pred_img contains NaN!")
        if torch.isnan(gt_img).any():
            raise ValueError("gt_img contains NaN!")

        # 1. 空间域损失 (Charbonnier Loss)
        loss_spa = self.charbonnier(pred_img, gt_img)

        # 2. 频域损失 (FFT Loss) - 添加数值稳定性
        pred_fft = torch.fft.rfft2(pred_img, norm='ortho')
        gt_fft = torch.fft.rfft2(gt_img, norm='ortho')
        # 分别计算实部和虚部的L1距离,使用 clamp 限制最大值
        loss_freq = torch.mean(torch.clamp(torch.abs(pred_fft.real - gt_fft.real), max=1000)) + \
                    torch.mean(torch.clamp(torch.abs(pred_fft.imag - gt_fft.imag), max=1000))

        # 3. 分类损失 (监督 DFAM 的判断能力)
        # 在交叉熵损失前添加 log softmax 保证数值稳定
        loss_cls = 0
        for stage_prob in pred_probs:
            # stage_prob 已经是 softmax 输出,直接计算
            # 添加小常数避免 log(0)
            log_prob = torch.log(stage_prob.clamp(min=1e-8, max=1.0))
            # 使用 NLL loss 替代 CrossEntropyLoss 以获得更好的数值稳定性
            nll_loss = F.nll_loss(log_prob, gt_labels)
            loss_cls += nll_loss
        loss_cls /= len(pred_probs)

        total_loss = loss_spa + self.lambda_freq * loss_freq + self.lambda_cls * loss_cls

        return total_loss, loss_spa, loss_freq, loss_cls

class L1_Charbonnier_Loss(nn.Module):
    """L1 Charbonnier Loss (针对图像恢复更鲁棒)"""
    def __init__(self, eps=1e-3):
        super(L1_Charbonnier_Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss
```

training/trainer.py
```python
import os
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from .losses import MultiTermLoss
from utils import TrainingLogger, MetricsTracker

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, logger=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config # 包含 lambda 等超参数的字典

        self.criterion = MultiTermLoss(
            lambda_freq=config.get('lambda_freq', 0.1),
            lambda_cls=config.get('lambda_cls', 0.05)
        )
        self.scaler = GradScaler("cuda") # 自动混合精度
        self.best_psnr = 0.0

        # 初始化日志系统
        if logger is None:
            self.logger = TrainingLogger(log_dir=config.get('log_dir', './logs'))
        else:
            self.logger = logger

        # 记录配置
        self.logger.log_config(config)

    def train_epoch(self, epoch):
        self.model.train()
        metrics_tracker = MetricsTracker()

        # 记录epoch开始
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_epoch_start(epoch, self.config['epochs'], current_lr)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 数据解包
            img_degraded, img_gt, gt_labels, blur_kernels = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()

            # 使用混合精度加速
            with autocast("cuda"):
                output, all_probs = self.model(img_degraded)
                loss, l_spa, l_freq, l_cls = self.criterion(output, img_gt, all_probs, gt_labels)

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                self.logger.logger.error(f"NaN detected at epoch {epoch}, batch {batch_idx}!")
                self.logger.logger.error(f"img_degraded: min={img_degraded.min():.4f}, max={img_degraded.max():.4f}")
                self.logger.logger.error(f"img_gt: min={img_gt.min():.4f}, max={img_gt.max():.4f}")
                self.logger.logger.error(f"output: min={output.min():.4f}, max={output.max():.4f}")
                continue  # 跳过这个batch

            self.scaler.scale(loss).backward()

            # 梯度裁剪 (防止梯度爆炸)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新指标追踪
            metrics = {
                'loss': loss.item(),
                'loss_spa': l_spa.item(),
                'loss_freq': l_freq.item(),
                'loss_cls': l_cls.item()
            }
            metrics_tracker.update(metrics, n=img_degraded.size(0))

            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Spa': f"{l_spa.item():.4f}",
                'Freq': f"{l_freq.item():.4f}",
                'Cls': f"{l_cls.item():.4f}"
            })

            # 记录到日志
            self.logger.log_iter(
                epoch=epoch,
                iteration=batch_idx,
                total_iters=len(self.train_loader),
                metrics=metrics
            )

        # 学习率调度
        self.scheduler.step()

        # 记录epoch结束
        epoch_metrics = metrics_tracker.get_metrics()
        self.logger.log_epoch_end(epoch, epoch_metrics, phase='train')

        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        psnr_val = 0.0
        metrics_tracker = MetricsTracker()

        for batch in self.val_loader:
            img_degraded, img_gt, gt_labels, _ = [x.to(self.device) for x in batch]

            output, _ = self.model(img_degraded)

            # 计算 PSNR
            mse = torch.mean((output - img_gt)**2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_val += psnr.item()

            # 更新指标
            metrics_tracker.update({'psnr': psnr.item()}, n=img_degraded.size(0))

        avg_psnr = psnr_val / len(self.val_loader)
        val_metrics = metrics_tracker.get_metrics()

        # 记录验证结果
        self.logger.log_epoch_end(epoch, val_metrics, phase='val')

        # 保存最佳模型
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint("best_model.pth")
            self.logger.log_best_metric(epoch, 'PSNR', avg_psnr)

    def save_checkpoint(self, name):
        checkpoint_path = os.path.join(self.config['save_dir'], name)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr
        }, checkpoint_path)
        self.logger.logger.info(f"模型已保存: {checkpoint_path}")
```

utils/data_loader.py
```python
import os
import glob
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class Rain100LDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, is_train=True):
        """
        root_dir: Rain100L 文件夹路径
        patch_size: 训练时裁剪的大小
        is_train: 是否为训练模式
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.is_train = is_train
        
        # 获取有雨图片路径
        self.rainy_images = sorted(glob.glob(os.path.join(root_dir, 'rainy', 'rain-*.png')))
        # 获取对应的无雨图片路径 (对应 norain-xxx.png)
        self.gt_images = []
        for r_path in self.rainy_images:
            # 提取编号，例如从 rain-001.png 提取 001
            idx = os.path.basename(r_path).split('-')[-1]
            gt_path = os.path.join(root_dir, f'norain-{idx}')
            # 兼容处理用户提到的 norain_xxx.png 或 norain-xxx.png
            if not os.path.exists(gt_path):
                gt_path = os.path.join(root_dir, f'norain_{idx}')
            self.gt_images.append(gt_path)

    def __len__(self):
        return len(self.rainy_images)

    def transform(self, rainy, gt):
        # 训练模式：随机裁剪和翻转
        if self.is_train:
            # 手动实现随机裁剪
            width, height = rainy.size
            i = random.randint(0, height - self.patch_size)
            j = random.randint(0, width - self.patch_size)
            h = w = self.patch_size
            rainy = F.crop(rainy, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)

            # 随机水平翻转
            if random.random() > 0.5:
                rainy = F.hflip(rainy)
                gt = F.hflip(gt)
        else:
            # 测试模式：为了能让 Batch 训练，通常裁剪中心区域或填充
            # 这里简单采用中心裁剪
            rainy = F.center_crop(rainy, (self.patch_size, self.patch_size))
            gt = F.center_crop(gt, (self.patch_size, self.patch_size))

        # 转为 Tensor
        rainy = F.to_tensor(rainy)
        gt = F.to_tensor(gt)
        return rainy, gt

    def __getitem__(self, index):
        rainy_img = Image.open(self.rainy_images[index]).convert('RGB')
        gt_img = Image.open(self.gt_images[index]).convert('RGB')

        rainy_tensor, gt_tensor = self.transform(rainy_img, gt_img)

        # 针对 Rain100L 任务的特定标签：
        # 根据之前的设计，0 代表 Identity (去雨/去噪)，1 代表 Convolution (去模糊)
        task_label = torch.tensor(0).long()
        
        # 因为去雨不需要模糊核，提供一个虚拟的单位核 (1x1) 或全 0 占位
        dummy_kernel = torch.zeros((1, 11, 11)) 

        return rainy_tensor, gt_tensor, task_label, dummy_kernel

def get_dataloader(root_dir, batch_size=16, patch_size=256, is_train=True, num_workers=4):
    dataset = Rain100LDataset(root_dir, patch_size=patch_size, is_train=is_train)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    return dataloader
```

train.py
```python
import torch
from torch.utils.data import DataLoader
from models import DFA_DUN # 假设你的模型文件名
from training import Trainer
from utils import get_dataloader

def main():
    # 1. 超参数配置
    config = {
        'K': 8,
        'lr': 1e-4,
        'batch_size': 16,
        'epochs': 100,
        'lambda_freq': 0.1,
        'lambda_cls': 0.05,
        'save_dir': './checkpoints'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 实例化模型
    # 确保你的 DFA_DUN forward 函数会返回 (output, probs_list)
    model = DFA_DUN(K=config['K'])

    # 3. 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # 4. 数据加载
    train_loader = get_dataloader(root_dir='./data/Rain100L', batch_size=config['batch_size'], is_train=True)
    val_loader = get_dataloader(root_dir='./data/Rain100L', batch_size=1, is_train=False)

    # 5. 启动训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )

    for epoch in range(1, config['epochs'] + 1):
        trainer.train_epoch(epoch)
        if epoch % 10 == 0:
            trainer.validate(epoch)

if __name__ == "__main__":
    main()
```