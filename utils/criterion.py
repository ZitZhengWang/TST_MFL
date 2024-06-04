import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
import utils


class KLDivLoss(nn.Module):
    def __init__(self, T):
        super(KLDivLoss, self).__init__()
        self.T = T

    def forward(self, input, target):
        input = F.log_softmax(input/self.T, dim=-1)
        target = F.softmax(target/self.T, dim=-1)

        kl_loss = F.kl_div(input, target, reduction='batchmean') * (self.T**2)

        # loss = F.kl_div(input, target, size_average=False) * (self.T**2) / input.shape[0]

        return kl_loss


class DistillLoss(nn.Module):
    def __init__(self, KLDivLoss):
        super(DistillLoss, self).__init__()
        self.KLDivLoss = KLDivLoss

    def forward(self, g_logits, l_avg_logits):
        # 输入全局 和 局部的预测结果
        # [B, n_class]    [B, n_class]
        # 球
        y_target = 0.5 * (g_logits + l_avg_logits)    # [B, n_class]
        loss_distill = self.KLDivLoss(l_avg_logits, y_target) + self.KLDivLoss(g_logits, y_target)
        return loss_distill


class DiscrepancyLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiscrepancyLoss, self).__init__()

    def forward(self, l_logits, l_avg_logits):
        # [B, 36, n_class]    [B, n_class]
        n_batch, n_patch, n_classes = l_logits.size()
        l_avg_logits = l_avg_logits.unsqueeze(1).repeat(1, n_patch, 1)    # [B, 36, 100]

        # 归一化
        l_logits = F.normalize(l_logits, dim=-1)
        l_avg_logits = F.normalize(l_avg_logits, dim=-1)

        # [B, 36, 100] * [B, 36， 100] =# [B, 36]

        logits = (l_logits * l_avg_logits).sum(dim=-1).sum(dim=-1) / sqrt(n_patch) / sqrt(n_classes)

        return logits.mean()


