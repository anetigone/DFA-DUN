import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTermLoss(nn.Module):
    def __init__(self, lambda_freq=0.1, lambda_cls=0.05):
        super(MultiTermLoss, self).__init__()
        self.lambda_freq = lambda_freq
        self.lambda_cls = lambda_cls
        self.charbonnier = L1_Charbonnier_Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred_img, gt_img, pred_probs, gt_labels):
        """
        pred_img: 恢复的图像 [B, 3, H, W]
        gt_img: 干净的图像 [B, 3, H, W]
        pred_probs: DFAM 输出的分类概率, 列表形式 [stage1_prob, stage2_prob, ...]
        gt_labels: 退化类型标签 [B] (例如: 0为恒等, 1为卷积)
        """
        # 检查输入是否包含 NaN
        if torch.isnan(pred_img).any():
            raise ValueError("pred_img contains NaN!")
        if torch.isnan(gt_img).any():
            raise ValueError("gt_img contains NaN!")

        # 1. 空间域损失 (Charbonnier Loss)
        loss_spa = self.charbonnier(pred_img, gt_img)

        # 2. 频域损失 (FFT Loss) - 添加数值稳定性
        pred_fft = torch.fft.rfft2(pred_img, norm='ortho')
        gt_fft = torch.fft.rfft2(gt_img, norm='ortho')
        # 分别计算实部和虚部的L1距离,使用 clamp 限制最大值
        loss_freq = torch.mean(torch.clamp(torch.abs(pred_fft.real - gt_fft.real), max=1000)) + \
                    torch.mean(torch.clamp(torch.abs(pred_fft.imag - gt_fft.imag), max=1000))

        # 3. 分类损失 (监督 DFAM 的判断能力)
        # 在交叉熵损失前添加 log softmax 保证数值稳定
        loss_cls = 0
        for stage_prob in pred_probs:
            # stage_prob 已经是 softmax 输出,直接计算
            # 添加小常数避免 log(0)
            log_prob = torch.log(stage_prob.clamp(min=1e-8, max=1.0))
            # 使用 NLL loss 替代 CrossEntropyLoss 以获得更好的数值稳定性
            nll_loss = F.nll_loss(log_prob, gt_labels)
            loss_cls += nll_loss
        loss_cls /= len(pred_probs)

        total_loss = loss_spa + self.lambda_freq * loss_freq + self.lambda_cls * loss_cls

        return total_loss, loss_spa, loss_freq, loss_cls

class L1_Charbonnier_Loss(nn.Module):
    """L1 Charbonnier Loss (针对图像恢复更鲁棒)"""
    def __init__(self, eps=1e-3):
        super(L1_Charbonnier_Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss