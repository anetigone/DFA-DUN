import os
import time
import logging
import json
from pathlib import Path


class TrainingLogger:
    """
    综合训练日志系统
    功能:
    1. 控制台输出 (带颜色)
    2. 文件日志记录
    3. 指标历史保存 (JSON)
    """
    def __init__(self, log_dir='./logs', exp_name=None, create_exp_subdir=False):
        """
        Args:
            log_dir: 日志保存目录
            exp_name: 实验名称,如果为None则自动生成时间戳
            create_exp_subdir: 是否创建实验子目录 (默认False,直接使用log_dir)
        """
        self.log_dir = Path(log_dir)
        self.exp_name = exp_name or f"exp_{time.strftime('%Y%m%d_%H%M%S')}"

        # 根据参数决定是否创建实验子目录
        if create_exp_subdir:
            self.exp_dir = self.log_dir / self.exp_name
            self.exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.exp_dir = self.log_dir
            self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. 设置文件日志
        log_file = self.exp_dir / 'train.log'
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)

        # 清除已有的handlers
        self.logger.handlers.clear()

        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 2. 指标历史
        self.metrics_history = {
            'train': [],
            'val': []
        }

        # 3. 保存配置
        self.config_file = self.exp_dir / 'config.json'

        self.logger.info(f"实验目录: {self.exp_dir}")

    def log_config(self, config):
        """保存训练配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.logger.info(f"配置已保存: {self.config_file}")

    def log_epoch_start(self, epoch, total_epochs, lr=None):
        """记录epoch开始"""
        msg = f"\n{'='*60}"
        msg += f"\nEpoch {epoch}/{total_epochs}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        msg += f"\n{'='*60}"
        self.logger.info(msg)

    def log_epoch_end(self, epoch, metrics, phase='train'):
        """
        记录epoch结束
        Args:
            epoch: 当前epoch
            metrics: 字典,包含各种指标
            phase: 'train' 或 'val'
        """
        # 记录到文件
        self.metrics_history[phase].append({
            'epoch': epoch,
            **metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

        # 控制台输出
        if phase == 'train':
            msg = f"训练 - Loss: {metrics.get('loss', 0):.4f}"
            if 'psnr' in metrics:
                msg += f" | PSNR: {metrics['psnr']:.2f} dB"
            if 'loss_spa' in metrics:
                msg += f" (Spa: {metrics['loss_spa']:.4f}, Freq: {metrics['loss_freq']:.4f}, Cls: {metrics['loss_cls']:.4f})"
        else:
            msg = f"验证 - PSNR: {metrics.get('psnr', 0):.2f} dB"
            if 'ssim' in metrics:
                msg += f" | SSIM: {metrics['ssim']:.4f}"

        self.logger.info(msg)

        # 保存JSON历史
        self._save_metrics()

    def log_iter(self, epoch, iteration, total_iters, metrics):
        """
        记录训练迭代
        Args:
            epoch: 当前epoch
            iteration: 当前iteration
            total_iters: 总iteration数
            metrics: 字典,包含各种指标
        """
        # 只在特定步数记录,避免日志过多
        log_interval = max(1, total_iters // 10)  # 每个epoch记录10次

        if iteration % log_interval == 0 or iteration == total_iters - 1:
            msg = f"[{iteration}/{total_iters}] "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(msg)

    def log_best_metric(self, epoch, metric_name, metric_value):
        """记录最佳指标"""
        msg = f"\n{'='*60}"
        msg += f"\n★ 最佳 {metric_name}: {metric_value:.4f} (Epoch {epoch})"
        msg += f"\n{'='*60}"
        self.logger.info(msg)

    def log_images(self, tag_prefix, images_dict, epoch):
        """
        批量记录图像
        Args:
            tag_prefix: 标签前缀
            images_dict: {tag: image} 字典
            epoch: 当前epoch
        """
        for tag, image in images_dict.items():
            self.log_image(f"{tag_prefix}/{tag}", image, epoch)

    def close(self):
        """关闭所有日志"""
        self.logger.info("日志已关闭")

    def _save_metrics(self):
        """保存指标历史到JSON"""
        metrics_file = self.exp_dir / 'metrics_history.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

    def get_metrics(self, phase='train'):
        """获取指定阶段的指标历史"""
        return self.metrics_history.get(phase, [])


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """指标追踪器,用于追踪多个指标"""
    def __init__(self):
        self.meters = {}

    def update(self, metrics_dict, n=1):
        """
        更新指标
        Args:
            metrics_dict: {指标名: 值} 字典
            n: batch size
        """
        for name, value in metrics_dict.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter()
            self.meters[name].update(value, n)

    def get_metrics(self):
        """获取所有指标的平均值"""
        return {name: meter.avg for name, meter in self.meters.items()}

    def reset(self):
        """重置所有指标"""
        self.meters.clear()
