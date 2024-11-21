import torch
from torch import nn

class RNNAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(RNNAligner, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, features):
        """
        前向传播，通过RNN对齐特征。
3. 循环神经网络（Recurrent Neural Networks, RNN）
RNN 可以用于处理序列数据，并通过其内部状态来对齐不同模态的特征。特别是双向 RNN（Bi-RNN）可以捕捉到序列的上下文信息。
        Args:
            features (torch.Tensor): 输入特征，形状为 (batch_size, sequence_length, input_dim)。

        Returns:
            torch.Tensor: 对齐后的特征，形状为 (batch_size, sequence_length, input_dim)。
        """
        output, _ = self.rnn(features)
        aligned_features = self.linear(output)
        return aligned_features
