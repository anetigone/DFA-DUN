import os
import glob
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class Rain100LDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, is_train=True):
        """
        root_dir: Rain100L 文件夹路径
        patch_size: 训练时裁剪的大小
        is_train: 是否为训练模式
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.is_train = is_train
        
        # 获取有雨图片路径
        self.rainy_images = sorted(glob.glob(os.path.join(root_dir, 'rainy', 'rain-*.png')))
        # 获取对应的无雨图片路径 (对应 norain-xxx.png)
        self.gt_images = []
        for r_path in self.rainy_images:
            # 提取编号，例如从 rain-001.png 提取 001
            idx = os.path.basename(r_path).split('-')[-1]
            gt_path = os.path.join(root_dir, f'norain-{idx}')
            # 兼容处理用户提到的 norain_xxx.png 或 norain-xxx.png
            if not os.path.exists(gt_path):
                gt_path = os.path.join(root_dir, f'norain_{idx}')
            self.gt_images.append(gt_path)

    def __len__(self):
        return len(self.rainy_images)

    def transform(self, rainy, gt):
        # 训练模式：随机裁剪和翻转
        if self.is_train:
            # 手动实现随机裁剪
            width, height = rainy.size
            i = random.randint(0, height - self.patch_size)
            j = random.randint(0, width - self.patch_size)
            h = w = self.patch_size
            rainy = F.crop(rainy, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)

            # 随机水平翻转
            if random.random() > 0.5:
                rainy = F.hflip(rainy)
                gt = F.hflip(gt)
        else:
            # 测试模式：为了能让 Batch 训练，通常裁剪中心区域或填充
            # 这里简单采用中心裁剪
            rainy = F.center_crop(rainy, (self.patch_size, self.patch_size))
            gt = F.center_crop(gt, (self.patch_size, self.patch_size))

        # 转为 Tensor
        rainy = F.to_tensor(rainy)
        gt = F.to_tensor(gt)
        return rainy, gt

    def __getitem__(self, index):
        rainy_img = Image.open(self.rainy_images[index]).convert('RGB')
        gt_img = Image.open(self.gt_images[index]).convert('RGB')

        rainy_tensor, gt_tensor = self.transform(rainy_img, gt_img)

        # 针对 Rain100L 任务的特定标签：
        # 根据之前的设计，0 代表 Identity (去雨/去噪)，1 代表 Convolution (去模糊)
        task_label = torch.tensor(0).long()
        
        # 因为去雨不需要模糊核，提供一个虚拟的单位核 (1x1) 或全 0 占位
        dummy_kernel = torch.zeros((1, 11, 11)) 

        return rainy_tensor, gt_tensor, task_label, dummy_kernel

def get_dataloader(root_dir, batch_size=16, patch_size=256, is_train=True, num_workers=4):
    dataset = Rain100LDataset(root_dir, patch_size=patch_size, is_train=is_train)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    return dataloader