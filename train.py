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