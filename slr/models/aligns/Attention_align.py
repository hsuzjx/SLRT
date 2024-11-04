import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1.0 / (key_dim ** 0.5)
        self.linear = nn.Linear(query_dim + value_dim, query_dim)

    def forward(self, queries, keys, values):
        """
        Forward pass of the attention mechanism.

        Args:
            queries (torch.Tensor): Queries of shape (batch_size, query_seq_len, query_dim).
            keys (torch.Tensor): Keys of shape (batch_size, key_seq_len, key_dim).
            values (torch.Tensor): Values of shape (batch_size, key_seq_len, value_dim).

        Returns:
            torch.Tensor: Aligned features of shape (batch_size, query_seq_len, value_dim).
        """
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values)
        combined = torch.cat((context, queries), dim=-1)
        output = self.linear(combined)
        return output
