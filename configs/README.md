# 配置文件说明

本目录包含 DFA-DUN 的训练配置文件。

## 配置文件列表

### default.yaml
默认训练配置，适用于大多数场景。
- 学习率: 5e-5
- Batch Size: 8
- Epochs: 100
- 验证间隔: 每 10 个 epoch

### fast_train.yaml
快速训练配置，用于调试和快速实验。
- 学习率: 1e-4
- Batch Size: 16
- Epochs: 20
- 验证间隔: 每 5 个 epoch

### high_quality.yaml
高质量训练配置，用于最终训练以获得最佳性能。
- 学习率: 1e-5
- Batch Size: 4
- Epochs: 300
- 验证间隔: 每 5 个 epoch

## 使用方法

### 使用默认配置训练:
```bash
python train.py
```

### 使用指定配置文件训练:
```bash
python train.py --config configs/fast_train.yaml
```

### 从检查点恢复训练:
```bash
python train.py --config configs/default.yaml --resume checkpoints/best_model.pth
```

## 自定义配置

你可以复制现有配置文件并修改参数来创建自己的配置:

```bash
cp configs/default.yaml configs/my_config.yaml
# 然后编辑 my_config.yaml
```

## 配置参数说明

### model
- `K`: 退化类型数量

### training
- `lr`: 学习率
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `weight_decay`: 权重衰减 (L2 正则化)

### loss
- `lambda_freq`: 频域损失权重
- `lambda_cls`: 分类损失权重

### data
- `train_root`: 训练数据根目录
- `val_root`: 验证数据根目录
- `num_workers`: 数据加载线程数
- `pin_memory`: 是否将数据固定在内存中

### checkpoint
- `save_dir`: 检查点保存目录
- `save_interval`: 验证间隔 (epoch)

### logging
- `log_dir`: 日志保存目录
- `log_interval`: 日志记录间隔 (iteration)

### device
- `cuda`: 使用 GPU 训练
- `cpu`: 使用 CPU 训练
