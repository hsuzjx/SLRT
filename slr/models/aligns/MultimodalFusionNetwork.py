import torch
from torch import nn

class MultimodalFusionNetwork(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, output_dim):
        super(MultimodalFusionNetwork, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, visual_features, text_features):
        """
        前向传播，通过多模态融合对齐特征。
        5. 多模态融合网络（Multimodal Fusion Network）
多模态融合网络通过将不同模态的特征拼接或相加来对齐特征，然后通过一个全连接层或其他网络结构进行进一步处理

        Args:
            visual_features (torch.Tensor): 视觉特征，形状为 (batch_size, sequence_length, visual_dim)。
            text_features (torch.Tensor): 文本特征，形状为 (batch_size, sequence_length, text_dim)。

        Returns:
            torch.Tensor: 对齐后的特征，形状为 (batch_size, sequence_length, output_dim)。
        """
        visual_embed = self.visual_proj(visual_features)
        text_embed = self.text_proj(text_features)
        
        fused_features = torch.cat((visual_embed, text_embed), dim=-1)
        aligned_features = self.fusion_layer(fused_features)
        
        return aligned_features
