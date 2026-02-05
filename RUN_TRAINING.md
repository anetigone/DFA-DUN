# 训练脚本使用指南

本项目提供了两个训练脚本:

## 1. train.py (原始脚本)
原始的训练脚本,配置硬编码在代码中。

```bash
python train.py
```

## 2. train_with_config.py (推荐使用)
使用配置文件的训练脚本,更加灵活和可维护。

### 使用示例:

#### 使用默认配置训练:
```bash
python train_with_config.py
```

#### 使用快速训练配置 (调试用):
```bash
python train_with_config.py --config configs/fast_train.yaml
```

#### 使用高质量训练配置 (获得最佳性能):
```bash
python train_with_config.py --config configs/high_quality.yaml
```

#### 从检查点恢复训练:
```bash
python train_with_config.py --config configs/default.yaml --resume checkpoints/best_model.pth
```

## 配置文件说明

### configs/default.yaml
- 适合大多数场景的标准配置
- 学习率: 5e-5
- Batch Size: 8
- Epochs: 100

### configs/fast_train.yaml
- 用于调试和快速实验
- 训练轮数少 (20 epochs)
- 批次大小大 (16)
- 验证频繁 (每 5 个 epoch)

### configs/high_quality.yaml
- 用于最终训练以获得最佳性能
- 学习率小 (1e-5)
- 训练轮数多 (300 epochs)
- 批次大小小 (4)
- 损失权重调整

## 命令行参数

- `--config`: 配置文件路径 (默认: configs/default.yaml)
- `--resume`: 检查点路径,用于恢复训练 (默认: None)

## 创建自定义配置

复制现有配置文件并修改:

```bash
cp configs/default.yaml configs/my_config.yaml
# 编辑 my_config.yaml
```

然后使用:

```bash
python train_with_config.py --config configs/my_config.yaml
```

## 推荐工作流程

1. **调试阶段**: 使用 `fast_train.yaml` 快速验证代码
   ```bash
   python train_with_config.py --config configs/fast_train.yaml
   ```

2. **正常训练**: 使用 `default.yaml` 进行标准训练
   ```bash
   python train_with_config.py
   ```

3. **最终训练**: 使用 `high_quality.yaml` 获得最佳性能
   ```bash
   python train_with_config.py --config configs/high_quality.yaml
   ```

4. **恢复训练**: 如果训练中断,可以从检查点恢复
   ```bash
   python train_with_config.py --resume checkpoints/best_model.pth
   ```
