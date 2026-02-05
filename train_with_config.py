import torch
import argparse
import time
from pathlib import Path
from models import DFA_DUN
from training import Trainer
from utils import get_dataloader
from configs import load_config


def main():
    """
    使用配置文件的训练脚本

    使用示例:
    1. 使用默认配置训练:
       python train_with_config.py

    2. 使用指定配置文件训练:
       python train_with_config.py --config configs/fast_train.yaml

    3. 从检查点恢复训练:
       python train_with_config.py --config configs/default.yaml --resume results/exp_20240101_120000/checkpoints/best_model.pth

    4. 指定实验名称:
       python train_with_config.py --config configs/default.yaml --exp-name my_experiment
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DFA-DUN Training with Config File')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径 (默认: configs/default.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='实验名称 (默认: 自动生成时间戳)')
    args = parser.parse_args()

    # 1. 从配置文件加载超参数
    print(f"正在加载配置文件: {args.config}")
    config = load_config(args.config)
    print(f"配置加载成功!")
    print(f"训练轮数: {config['epochs']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['lr']}")

    # 2. 创建实验目录结构
    exp_name = args.exp_name or f"exp_{time.strftime('%Y%m%d_%H%M%S')}"

    results_dir = Path('./results')
    exp_dir = results_dir / exp_name
    checkpoints_dir = exp_dir / 'checkpoints'
    logs_dir = exp_dir / 'logs'

    # 创建所有必要的目录
    for dir_path in [results_dir, exp_dir, checkpoints_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 更新配置
    config.update({
        'save_dir': str(checkpoints_dir),
        'log_dir': str(logs_dir),
        'exp_name': exp_name,
        'exp_dir': str(exp_dir)
    })

    print(f"\n实验目录结构:")
    print(f"  实验根目录: {exp_dir}")
    print(f"  检查点位置: {checkpoints_dir}")
    print(f"  日志位置: {logs_dir}")
    print()

    # 设置设备
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"使用设备: {device}")

    # 2. 实例化模型
    model = DFA_DUN(K=config['K'])
    print(f"模型初始化完成 (K={config['K']})")

    # 3. 优化器与调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    print("优化器和调度器初始化完成")

    # 4. 数据加载
    data_config = config.get('data_config', {})
    train_root = data_config.get('train_root', './data/Rain100L')
    val_root = data_config.get('val_root', './data/Rain100L')

    print(f"加载训练数据: {train_root}")
    train_loader = get_dataloader(
        root_dir=train_root,
        batch_size=config['batch_size'],
        is_train=True
    )

    print(f"加载验证数据: {val_root}")
    val_loader = get_dataloader(
        root_dir=val_root,
        batch_size=1,
        is_train=False
    )
    print(f"数据加载完成 (训练集: {len(train_loader)} 批次, 验证集: {len(val_loader)} 批次)")

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

    # 恢复训练（如果指定）
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.best_psnr = checkpoint.get('best_psnr', 0.0)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"从检查点恢复训练: {args.resume}")
        print(f"最佳 PSNR: {trainer.best_psnr:.4f}")

    # 训练循环
    save_interval = config.get('save_interval', 10)
    print(f"\n开始训练 (验证间隔: 每 {save_interval} 个 epoch)")
    print("=" * 80)

    for epoch in range(start_epoch, config['epochs'] + 1):
        trainer.train_epoch(epoch)
        if epoch % save_interval == 0:
            trainer.validate(epoch)

    print("\n训练完成!")
    print(f"最佳 PSNR: {trainer.best_psnr:.4f}")


if __name__ == "__main__":
    main()
