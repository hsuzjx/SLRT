import torch
from torch import nn

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        """
        计算KL散度损失。

        Args:
            pred (torch.Tensor): 预测的概率分布，形状为 (batch_size, num_classes)。
            target (torch.Tensor): 目标概率分布，形状为 (batch_size, num_classes)。

        Returns:
            torch.Tensor: KL散度损失值。
        """
        pred_log_prob = torch.log_softmax(pred, dim=-1)
        target_prob = torch.softmax(target, dim=-1)
        loss = self.kl_div(pred_log_prob, target_prob)
        return loss
