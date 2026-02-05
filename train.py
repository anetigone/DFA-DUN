import torch
from torch.utils.data import DataLoader
from models import DFA_DUN # 假设你的模型文件名
from training import Trainer
from utils import get_dataloader
import time
from pathlib import Path

def main():
    # 1. 超参数配置
    exp_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"  # 实验名称

    # 创建实验目录结构
    results_dir = Path('./results')
    exp_dir = results_dir / exp_name
    checkpoints_dir = exp_dir / 'checkpoints'
    logs_dir = exp_dir / 'logs'

    # 创建所有必要的目录
    for dir_path in [results_dir, exp_dir, checkpoints_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    config = {
        'K': 4,
        'lr': 5e-5,
        'batch_size': 4,
        'epochs': 10,
        'lambda_freq': 0.1,
        'lambda_cls': 0.05,
        'save_dir': str(checkpoints_dir),  # 修改为 checkpoints 子目录
        'log_dir': str(logs_dir),         # 添加日志目录
        'exp_name': exp_name,             # 实验名称
        'exp_dir': str(exp_dir)           # 实验根目录
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"实验目录: {exp_dir}")
    print(f"检查点保存位置: {checkpoints_dir}")
    print(f"日志保存位置: {logs_dir}")

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