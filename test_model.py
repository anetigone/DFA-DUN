"""
DFA-DUN 模型测试脚本
用于加载训练好的模型并进行推理测试
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
import sys

# 导入模型相关模块
from models.dfa_dun import DFA_DUN
from utils.data_loader import Rain100LDataset


def load_model(checkpoint_path, K=4, device='cuda'):
    """
    加载训练好的模型

    Args:
        checkpoint_path: 模型检查点文件路径
        K: 迭代次数
        device: 运行设备

    Returns:
        model: 加载好的模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    print(f"正在加载模型: {checkpoint_path}")

    # 初始化模型
    model = DFA_DUN(K=K).to(device)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if 'best_psnr' in checkpoint:
            print(f"训练时的最佳PSNR: {checkpoint['best_psnr']:.4f} dB")
    else:
        state_dict = checkpoint

    # 加载权重到模型
    model.load_state_dict(state_dict)
    model.eval()

    print("模型加载成功！")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    return model


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_ssim(img1, img2):
    """计算SSIM (简化版)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.std()
    sigma2 = img2.std()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))

    return ssim.item()


@torch.no_grad()
def test_single_image(model, image_path, device='cuda'):
    """对单张图片进行推理"""
    img = Image.open(image_path).convert('RGB')
    input_tensor = to_tensor(img).unsqueeze(0).to(device)

    print(f"输入图片尺寸: {input_tensor.shape}")

    # 模型推理
    output, probs = model(input_tensor)

    return output, input_tensor, probs


def visualize_and_save(input_tensor, output_tensor, gt_tensor=None, save_path=None):
    """可视化并保存结果"""
    # 转换为numpy数组
    input_img = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_img = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # 确保范围在[0, 1]
    input_img = np.clip(input_img, 0, 1)
    output_img = np.clip(output_img, 0, 1)

    num_images = 2 if gt_tensor is None else 3
    fig, axes = plt.subplots(1, num_images, figsize=(6 * num_images, 6))

    axes[0].imshow(input_img)
    axes[0].set_title('Input (Rainy)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(output_img)
    axes[1].set_title('Output (Derained)', fontsize=14)
    axes[1].axis('off')

    if gt_tensor is not None:
        gt_img = gt_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        gt_img = np.clip(gt_img, 0, 1)

        axes[2].imshow(gt_img)
        axes[2].set_title('Ground Truth', fontsize=14)
        axes[2].axis('off')

        # 计算指标
        psnr = calculate_psnr(torch.from_numpy(output_img).permute(2, 0, 1),
                             torch.from_numpy(gt_img).permute(2, 0, 1))
        fig.suptitle(f'PSNR: {psnr:.2f} dB', fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"可视化结果已保存至: {save_path}")

    plt.show()


@torch.no_grad()
def evaluate_dataset(model, dataset, device='cuda', save_results=True, output_dir='./test_results'):
    """在整个测试集上评估模型"""
    model.eval()

    psnr_list = []
    ssim_list = []

    # 创建输出子目录
    if save_results:
        output_img_dir = os.path.join(output_dir, 'output_images')
        os.makedirs(output_img_dir, exist_ok=True)

    print(f"开始评估，共 {len(dataset)} 张图片...")

    for idx in tqdm(range(len(dataset))):
        # 获取数据
        rainy_tensor, gt_tensor, _, _ = dataset[idx]
        rainy_tensor = rainy_tensor.unsqueeze(0).to(device)
        gt_tensor = gt_tensor.unsqueeze(0).to(device)

        # 推理
        output, _ = model(rainy_tensor)

        # 计算PSNR和SSIM
        psnr = calculate_psnr(output, gt_tensor)
        ssim = calculate_ssim(output, gt_tensor)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        # 保存结果图片
        if save_results:
            output_img = to_pil_image(output.squeeze(0).cpu())
            save_path = os.path.join(output_img_dir, f'output_{idx+1:03d}.png')
            output_img.save(save_path)

    # 计算平均指标
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    metrics = {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'psnr_list': psnr_list,
        'ssim_list': ssim_list
    }

    print(f"\n评估完成！")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")

    return metrics


def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 请修改以下路径为你的实际路径
    CHECKPOINT_PATH = './results/exp_20250105_120000/checkpoints/best_model.pth'
    TEST_DATA_DIR = './data/Rain100L'
    OUTPUT_DIR = './test_results'

    # 模型参数
    K = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 打印配置信息
    print(f"设备: {DEVICE}")
    print(f"检查点路径: {CHECKPOINT_PATH}")
    print(f"测试数据路径: {TEST_DATA_DIR}")
    print(f"输出路径: {OUTPUT_DIR}\n")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    model = load_model(CHECKPOINT_PATH, K=K, device=DEVICE)

    # 2. 单张图片测试
    print("\n" + "="*50)
    print("单张图片测试")
    print("="*50)

    test_image_path = os.path.join(TEST_DATA_DIR, 'rainy', 'rain-001.png')
    gt_image_path = os.path.join(TEST_DATA_DIR, 'norain', 'norain-001.png')

    if os.path.exists(test_image_path):
        output, input_tensor, probs = test_single_image(model, test_image_path, DEVICE)

        # 如果有ground truth，加载并可视化
        if os.path.exists(gt_image_path):
            gt_img = Image.open(gt_image_path).convert('RGB')
            gt_tensor = to_tensor(gt_img).unsqueeze(0).to(DEVICE)
            save_path = os.path.join(OUTPUT_DIR, 'visualization_result.png')
            visualize_and_save(input_tensor, output, gt_tensor, save_path)
        else:
            save_path = os.path.join(OUTPUT_DIR, 'visualization_result.png')
            visualize_and_save(input_tensor, output, save_path=save_path)
    else:
        print(f"测试图片不存在: {test_image_path}")

    # 3. 批量评估
    print("\n" + "="*50)
    print("批量评估")
    print("="*50)

    test_dataset = Rain100LDataset(root_dir=TEST_DATA_DIR, patch_size=256, is_train=False)
    print(f"测试数据集大小: {len(test_dataset)}")

    metrics = evaluate_dataset(model, test_dataset, DEVICE, save_results=True, output_dir=OUTPUT_DIR)

    # 4. 保存评估报告
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("DFA-DUN 模型评估报告\n")
        f.write("="*50 + "\n\n")

        f.write(f"模型检查点: {CHECKPOINT_PATH}\n")
        f.write(f"测试数据集: {TEST_DATA_DIR}\n")
        f.write(f"测试样本数: {len(test_dataset)}\n")
        f.write(f"设备: {DEVICE}\n\n")

        f.write("="*50 + "\n")
        f.write("评估结果\n")
        f.write("="*50 + "\n\n")

        f.write(f"平均 PSNR: {metrics['avg_psnr']:.4f} ± {np.std(metrics['psnr_list']):.4f} dB\n")
        f.write(f"平均 SSIM: {metrics['avg_ssim']:.4f} ± {np.std(metrics['ssim_list']):.4f}\n\n")

        f.write("="*50 + "\n")
        f.write("详细结果 (前20张)\n")
        f.write("="*50 + "\n\n")

        for i in range(min(20, len(metrics['psnr_list']))):
            f.write(f"图片 {i+1:3d}: PSNR = {metrics['psnr_list'][i]:.4f} dB, "
                   f"SSIM = {metrics['ssim_list'][i]:.4f}\n")

        f.write("\n" + "="*50 + "\n")
        f.write("报告生成完成\n")
        f.write("="*50 + "\n")

    print(f"\n评估报告已保存至: {report_path}")

    # 绘制指标分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PSNR分布
    axes[0].hist(metrics['psnr_list'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(metrics['avg_psnr'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {metrics['avg_psnr']:.2f}")
    axes[0].set_xlabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('PSNR Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # SSIM分布
    axes[1].hist(metrics['ssim_list'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(metrics['avg_ssim'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {metrics['avg_ssim']:.3f}")
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('SSIM Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
