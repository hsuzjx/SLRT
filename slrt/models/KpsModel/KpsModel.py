from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from .BaseModel import BaseModel
from .modules import BiLSTMLayer, SeqKD
from .modules.Identity import Identity
from .modules.STGCN import STGCN


class KpsModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'KpsModel'

        self.kps_dict = {
            'body': [i for i in range(0, 11)],
            'left_hand': [i for i in range(91, 112)],
            'right_hand': [i for i in range(112, 133)],
            'face': [i for i in range(23, 40)],
            'mouth': [i for i in range(71, 91)],
            'nose': [i for i in range(50, 59)],
            'eyes': [i for i in range(59, 71)],
            'eyebrows': [i for i in range(40, 50)]
        }

    @override
    def _init_network(self, **kwargs):
        # self.gcn_body = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={},
        #     edge_importance_weighting=True,
        #     num_nodes=9
        # )
        # self.gcn_body.fcn = Identity()
        # self.gcn_hand = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={},
        #     edge_importance_weighting=True,
        #     num_nodes=21
        # )
        # self.gcn_hand.fcn = Identity()
        # self.gcn_face = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={},
        #     edge_importance_weighting=True,
        #     num_nodes=68
        # )
        # self.gcn_face.fcn = Identity()
        # self.mlp = nn.Sequential(
        #     nn.Linear(256 * 4, 1024),
        #     # nn.LayerNorm(1024, eps=1e-6),
        #     nn.ReLU(),
        # )

        self.gcn = STGCN(
            in_channels=3,
            num_class=1000,
            graph_args={},
            edge_importance_weighting=True,
            num_nodes=121
        )
        self.gcn.fcn = Identity()
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.ReLU(),
        )

        # self.conv1d = TemporalConv(
        #     input_size=256,
        #     hidden_size=512,
        #     conv_type=2,
        #     use_bn=True
        # )

        self.lstm = BiLSTMLayer(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            rnn_type='LSTM'
        )

        self.classifier = NormLinear(512, self.recognition_tokenizer.vocab_size)

    @override
    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            blank=0,
            reduction='none',
            zero_infinity=False
        )
        self.dist_loss = SeqKD(
            T=8
        )

    @override
    def forward(self, kps, kps_lgt) -> Any:
        # 获取输入数据的维度
        N, C, T, V = kps.shape
        # kps = kps[:, :2, :, :]
        kps = kps.unsqueeze(-1)
        N, C, T, V, M = kps.shape

        kps_body = kps[:, :, :, self.kps_dict['body'], :]
        kps_left_hand = kps[:, :, :, self.kps_dict['left_hand'], :]
        kps_right_hand = kps[:, :, :, self.kps_dict['right_hand'], :]
        kps_face = kps[:, :, :,
                   self.kps_dict['face'] + self.kps_dict['eyes'] + self.kps_dict['eyebrows'] + self.kps_dict['mouth'] +
                   self.kps_dict['nose'],
                   :]

        # f_body = self.gcn_body(kps_body)
        # f_left_hand = self.gcn_hand(kps_left_hand)
        # f_right_hand = self.gcn_hand(kps_right_hand)
        # f_face = self.gcn_face(kps_face)

        # f_concat = torch.cat((f_body, f_left_hand, f_right_hand, f_face), dim=1)
        f_concat = self.gcn(kps[:, :, :,
                   self.kps_dict['body']+
                   self.kps_dict['left_hand'] +
                   self.kps_dict['right_hand'] +
                   self.kps_dict['face'] +
                   self.kps_dict['eyes'] +
                   self.kps_dict['eyebrows'] +
                   self.kps_dict['mouth'] +
                   self.kps_dict['nose'],
                   :])
        f_fuse = self.mlp(f_concat.permute(0, 2, 1))

        # f_conv1d = self.conv1d(f_fuse)

        f_fuse = f_fuse.permute(1, 0, 2)  # ntc -> (T, N, C)
        v_len = kps_lgt // 4
        f_lstm, _ = self.lstm(f_fuse, v_len.cpu())

        return self.classifier(f_fuse), self.classifier(f_lstm), v_len

    @override
    def step_forward(self, batch) -> Any:
        kps, y_glosses, y_translation, kps_lgt, y_glosses_lgt, y_translation_lgt, name = batch
        batch_size = len(name)

        outputs = self.forward(kps, kps_lgt)

        if self.trainer.predicting:
            return torch.tensor([]), outputs[1], None, outputs[2], None, name

        loss = (
                1 * self.ctc_loss(outputs[0].log_softmax(-1), y_glosses, outputs[2].cpu(), y_glosses_lgt.cpu()).mean() +
                1 * self.ctc_loss(outputs[1].log_softmax(-1), y_glosses, outputs[2].cpu(), y_glosses_lgt.cpu()).mean() +
                25 * self.dist_loss(outputs[0], outputs[1].detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        if "translation" in self.task:
            pass

        return loss, outputs[1], None, outputs[2], None, name


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
