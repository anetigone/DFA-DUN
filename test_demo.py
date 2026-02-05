import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from models.dfa_dun import DFA_DUN
from utils.data_loader import Rain100LDataset
import glob

def load_model(checkpoint_path, device, K=4):
    """加载训练好的模型"""
    model = DFA_DUN(K=K).to(device)

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理不同的checkpoint格式
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint with best PSNR: {checkpoint.get('best_psnr', 'N/A')}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def tensor_to_image(tensor):
    """将tensor转换为PIL Image"""
    # tensor: [C, H, W], 范围 [0, 1]
    tensor = tensor.clamp(0, 1)
    image = F.to_pil_image(tensor)
    return image

def save_comparison(rainy, output, gt, save_path, idx):
    """保存对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Rainy image
    axes[0].imshow(rainy)
    axes[0].set_title('Rainy Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Output (Restored)
    axes[1].imshow(output)
    axes[1].set_title('Restored (DFA-DUN)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(gt)
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {save_path}")

def calculate_metrics(output, gt):
    """计算PSNR和SSIM"""
    from sklearn.metrics import mean_squared_error
    import math

    # 转为numpy数组
    output_np = output.cpu().numpy().transpose(1, 2, 0)
    gt_np = gt.cpu().numpy().transpose(1, 2, 0)

    # 计算MSE
    mse = np.mean((output_np - gt_np) ** 2)

    # 计算PSNR
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))

    return psnr

def test_single_images(model, device, data_dir, num_samples=10, save_dir='results'):
    """测试单张图片并保存结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 获取测试图片
    rainy_images = sorted(glob.glob(os.path.join(data_dir, 'rainy', 'rain-*.png')))[:num_samples]

    print(f"\nTesting {len(rainy_images)} images...")
    print("=" * 60)

    total_psnr = 0

    with torch.no_grad():
        for idx, rainy_path in enumerate(rainy_images):
            # 获取文件名
            basename = os.path.basename(rainy_path)
            file_idx = basename.split('-')[-1]
            gt_path = os.path.join(data_dir, f'norain-{file_idx}')

            # 读取图片
            rainy_img = Image.open(rainy_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')

            # 中心裁剪以匹配训练时的尺寸
            rainy_cropped = F.center_crop(rainy_img, (256, 256))
            gt_cropped = F.center_crop(gt_img, (256, 256))

            # 转为tensor
            rainy_tensor = F.to_tensor(rainy_cropped).unsqueeze(0).to(device)  # [1, 3, 256, 256]
            gt_tensor = F.to_tensor(gt_cropped).to(device)

            # 模型推理
            output, probs = model(rainy_tensor)

            # 转换回图像
            output_img = tensor_to_image(output[0].cpu())

            # 保存对比图
            save_path = os.path.join(save_dir, f'comparison_{idx+1:03d}.png')
            save_comparison(rainy_cropped, output_img, gt_cropped, save_path, idx)

            # 计算PSNR
            psnr = calculate_metrics(output[0], gt_tensor)
            total_psnr += psnr

            # 打印第一个stage的gate概率
            print(f"\n[{idx+1}/{len(rainy_images)}] {basename}")
            print(f"  PSNR: {psnr:.2f} dB")
            if len(probs) > 0:
                gate_probs = probs[0][0].cpu().numpy()  # [4]
                print(f"  Gate probs: Identity={gate_probs[0]:.3f}, Blur={gate_probs[1]:.3f}, "
                      f"Noise={gate_probs[2]:.3f}, Loss={gate_probs[3]:.3f}")

    avg_psnr = total_psnr / len(rainy_images)
    print("\n" + "=" * 60)
    print(f"Average PSNR: {avg_psnr:.2f} dB over {len(rainy_images)} images")
    print("=" * 60)

def test_with_dataloader(model, device, data_dir, batch_size=1, num_batches=10, save_dir='results'):
    """使用dataloader测试"""
    os.makedirs(save_dir, exist_ok=True)

    # 创建数据集
    dataset = Rain100LDataset(data_dir, patch_size=256, is_train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTesting with dataloader ({min(num_batches, len(dataloader))} batches)...")
    print("=" * 60)

    total_psnr = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (rainy, gt, task_label, dummy_kernel) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            rainy = rainy.to(device)
            gt = gt.to(device)

            # 模型推理
            output, probs = model(rainy)

            # 保存结果
            for i in range(rainy.size(0)):
                save_path = os.path.join(save_dir, f'batch_{batch_idx+1}_sample_{i+1}.png')

                rainy_img = tensor_to_image(rainy[i].cpu())
                output_img = tensor_to_image(output[i].cpu())
                gt_img = tensor_to_image(gt[i].cpu())

                save_comparison(rainy_img, output_img, gt_img, save_path, batch_idx * batch_size + i)

                # 计算PSNR
                psnr = calculate_metrics(output[i], gt[i])
                total_psnr += psnr
                count += 1

                # 打印gate概率
                if len(probs) > 0:
                    gate_probs = probs[0][i].cpu().numpy()
                    print(f"\n[Batch {batch_idx+1}, Sample {i+1}]")
                    print(f"  PSNR: {psnr:.2f} dB")
                    print(f"  Gate probs: Identity={gate_probs[0]:.3f}, Blur={gate_probs[1]:.3f}, "
                          f"Noise={gate_probs[2]:.3f}, Loss={gate_probs[3]:.3f}")

    avg_psnr = total_psnr / count if count > 0 else 0
    print("\n" + "=" * 60)
    print(f"Average PSNR: {avg_psnr:.2f} dB over {count} images")
    print("=" * 60)

def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = './results/exp_20260205_152700/checkpoints/best_model.pth'
    data_dir = './data/Rain100L'
    save_dir = './test_results'
    K = 4  # 迭代次数，需要与训练时一致

    print("=" * 60)
    print("DFA-DUN Image Restoration Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"K (iterations): {K}")
    print("=" * 60)

    # 检查checkpoint是否存在
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # 加载模型
    print("\nLoading model...")
    model = load_model(checkpoint_path, device, K=K)
    print("Model loaded successfully!")

    # 方式1: 直接处理单张图片 (推荐，可以看到原始图片大小)
    test_single_images(model, device, data_dir, num_samples=10, save_dir=os.path.join(save_dir, 'single_images'))

    # 方式2: 使用dataloader (使用训练时的数据预处理)
    # test_with_dataloader(model, device, data_dir, batch_size=1, num_batches=10, save_dir=os.path.join(save_dir, 'dataloader'))

    print(f"\nResults saved to: {save_dir}")

if __name__ == "__main__":
    main()
