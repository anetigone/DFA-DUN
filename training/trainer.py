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