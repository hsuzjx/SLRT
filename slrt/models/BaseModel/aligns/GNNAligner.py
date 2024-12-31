import torch
from torch import nn
from torch_geometric.nn import GCNConv

class GNNAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNAligner, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        """
        前向传播，通过GNN对齐特征。
4. 图神经网络（Graph Neural Networks, GNN）
GNN 可以用于建模不同模态之间的关系图，通过图卷积操作来对齐特征。
        Args:
            x (torch.Tensor): 输入特征，形状为 (num_nodes, input_dim)。
            edge_index (torch.Tensor): 边的索引，形状为 (2, num_edges)。

        Returns:
            torch.Tensor: 对齐后的特征，形状为 (num_nodes, input_dim)。
        """
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return x
