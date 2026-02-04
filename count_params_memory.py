"""
计算DFA-DUN模型的参数量和训练显存需求
"""
import torch
import torch.nn as nn
from models.dfa_dun import DFA_DUN

def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"模型参数统计")
    print(f"{'='*60}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数量(M): {total_params/1e6:.2f}M")

    # 详细统计各模块参数
    print(f"\n{'='*60}")
    print(f"各模块参数详情")
    print(f"{'='*60}")

    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name:30s}: {module_params:>12,} ({module_params/1e6:.2f}M)")

    return total_params

def estimate_memory(model, input_size=(3, 256, 256), batch_size=16):
    """
    估算训练显存占用

    显存占用组成:
    1. 模型权重
    2. 优化器状态 (AdamW: 2个状态 * 权重)
    3. 前向激活值 (约等于权重大小)
    4. 梯度 (等于权重大小)
    5. 输入数据
    6. 中间变量和临时缓冲区
    """
    print(f"\n{'='*60}")
    print(f"显存占用估算 (batch_size={batch_size}, img_size={input_size})")
    print(f"{'='*60}")

    # 1. 模型权重 (FP32)
    total_params = sum(p.numel() for p in model.parameters())
    model_memory = total_params * 4 / (1024**3)  # 4 bytes per float32

    # 2. 优化器状态 (AdamW: 2个状态 * 权重)
    optimizer_memory = model_memory * 2

    # 3. 梯度
    gradient_memory = model_memory

    # 4. 前向激活值 (粗略估计:约等于权重大小,取决于网络深度)
    activation_memory = model_memory * 1.5

    # 5. 输入数据
    c, h, w = input_size
    input_per_sample = c * h * w * 4 / (1024**3)  # float32
    input_memory = input_per_sample * batch_size

    # 6. 标签/目标
    label_memory = input_memory

    # 7. 额外开销 (中间变量、FFT缓冲区等)
    overhead = (model_memory + activation_memory) * 0.3

    total_memory = (
        model_memory +
        optimizer_memory +
        gradient_memory +
        activation_memory +
        input_memory +
        label_memory +
        overhead
    )

    print(f"\n组成:")
    print(f"  模型权重 (FP32):          {model_memory:.2f} GB")
    print(f"  优化器状态 (AdamW x2):    {optimizer_memory:.2f} GB")
    print(f"  梯度:                    {gradient_memory:.2f} GB")
    print(f"  前向激活值:              {activation_memory:.2f} GB")
    print(f"  输入数据 (batch={batch_size}):     {input_memory:.2f} GB")
    print(f"  标签数据:                {label_memory:.2f} GB")
    print(f"  额外开销:                {overhead:.2f} GB")
    print(f"{'-'*60}")
    print(f"  总计:                    {total_memory:.2f} GB")

    # 不同batch size下的预估
    print(f"\n不同batch size下的显存需求:")
    print(f"{'-'*60}")
    for bs in [1, 2, 4, 8, 16, 32]:
        # 可变部分
        var_input = input_per_sample * bs * 2  # input + label
        var_activation = activation_memory * (bs / batch_size)
        var_overhead = overhead * (bs / batch_size)

        total_var = (
            model_memory +
            optimizer_memory +
            gradient_memory +
            var_activation +
            var_input +
            var_overhead
        )
        print(f"  batch_size={bs:2d}:  {total_var:>6.2f} GB")

    return total_memory

def detailed_layer_analysis(model):
    """详细分析每层的参数"""
    print(f"\n{'='*60}")
    print(f"逐层参数分析")
    print(f"{'='*60}")
    print(f"{'Layer Name':<40} {'Shape':>20} {'Params':>12}")
    print(f"{'-'*60}")

    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = str(list(param.shape))
            num_params = param.numel()
            total += num_params
            print(f"{name:<40} {shape:>20} {num_params:>12,}")

    print(f"{'-'*60}")
    print(f"{'Total':<61} {total:>12,}")

if __name__ == "__main__":
    # 创建模型
    print("正在创建模型...")
    model = DFA_DUN(K=8, in_nc=3)

    # 统计参数
    total_params = count_parameters(model)

    # 估算显存
    estimate_memory(model, input_size=(3, 256, 256), batch_size=16)

    # 详细分析
    detailed_layer_analysis(model)

    print(f"\n{'='*60}")
    print("建议:")
    print(f"{'='*60}")
    print("1. 如果显存不足,可以尝试:")
    print("   - 减小 batch_size (如 8, 4, 2)")
    print("   - 使用梯度累积")
    print("   - 使用混合精度训练 (FP16) 可节省约40%显存")
    print("   - 使用梯度检查点 (gradient checkpointing)")
    print(f"\n2. 混合精度训练预估显存:")
    mixed_precision_memory = estimate_memory(model, input_size=(3, 256, 256), batch_size=16) * 0.6
    print(f"   batch_size=16:  ~{mixed_precision_memory:.2f} GB")
