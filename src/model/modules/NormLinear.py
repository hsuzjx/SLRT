import torch
import torch.nn.functional as F
from torch import nn


class NormLinear(nn.Module):
    """
    实现规范化线性层，继承自nn.Module。
    
    该层的主要功能是对输入进行线性变换，其中线性变换的权重被规范化。
    支持添加偏置项，但默认不添加。
    
    Attributes:
        weight (nn.Parameter): 线性变换的权重参数。
        bias (nn.Parameter, optional): 偏置项参数，仅在add_bias为True时存在。
    """

    def __init__(self, in_dim, out_dim, add_bias=False):
        """
        初始化NormLinear层。
        
        Parameters:
            in_dim (int): 输入特征维度。
            out_dim (int): 输出特征维度。
            add_bias (bool, optional): 是否添加偏置项，默认为False。
        """
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        self.add_bias = add_bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))  # 添加偏置项
            nn.init.zeros_(self.bias)  # 初始化偏置项为零

    def forward(self, x):
        """
        对输入张量与规范化后的权重进行矩阵乘法操作，并加上偏置项。
        
        Parameters:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, feature_dim)。
        
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, out_dim)。
        """
        # 验证输入张量的形状
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be 3D, got shape {x.shape}")
        batch_size, sequence_length, feature_dim = x.shape
        if feature_dim != self.weight.shape[0]:
            raise ValueError("Feature dimension of input does not match weight")

        # 规范化权重
        normalized_weight = F.normalize(self.weight, dim=0)

        # 矩阵乘法
        outputs = torch.matmul(x, normalized_weight)

        if self.add_bias:
            # 添加偏置项
            outputs += self.bias  # 偏置项会自动广播以匹配输出的形状

        return outputs
