import torch
from torch import nn

class CrossModalMatchingNetwork(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super(CrossModalMatchingNetwork, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, visual_features, text_features):
        """
        前向传播，计算视觉特征和文本特征的相似度。
2. 交叉模态匹配网络（Cross-Modal Matching Network, CMMN）
CMMN 通过学习一个共享的嵌入空间来对齐不同模态的特征。这种方法通常用于多模态任务，如图像-文本匹配
        Args:
            visual_features (torch.Tensor): 视觉特征，形状为 (batch_size, sequence_length, visual_dim)。
            text_features (torch.Tensor): 文本特征，形状为 (batch_size, sequence_length, text_dim)。

        Returns:
            torch.Tensor: 相似度矩阵，形状为 (batch_size, sequence_length, sequence_length)。
        """
        visual_embed = self.visual_proj(visual_features)
        text_embed = self.text_proj(text_features)
        
        # 计算相似度矩阵
        similarity_matrix = self.cosine_sim(visual_embed.unsqueeze(2), text_embed.unsqueeze(1))
        
        return similarity_matrix
