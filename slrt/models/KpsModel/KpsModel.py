from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from .BaseModel import BaseModel
from .modules import BiLSTMLayer, SeqKD, AdjacencyLearn, DynamicLSTM
from .modules.DeGCN import DeGCN
from .modules.Identity import Identity
from .modules.STGCN import STGCN


class KpsModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'KpsModel'

    @override
    def _init_network(self, **kwargs):
        self.kps_dict = {
            'body': [i for i in range(0, 11)],
            'left_hand': [i for i in range(91, 112)],
            'right_hand': [i for i in range(112, 133)],
            'all_face': [i for i in range(23, 91)],
            'face': [i for i in range(23, 40)],
            'mouth': [i for i in range(71, 91)],
            'nose': [i for i in range(50, 59)],
            'eyes': [i for i in range(59, 71)],
            'eyebrows': [i for i in range(40, 50)]
        }
        # self.kps_dict = {
        #     "body": [
        #         0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        #         100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        #         111, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118,
        #         119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
        #         130, 131, 132, 23, 26, 29, 33, 36, 39, 41, 43, 46, 48,
        #         53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79,
        #         80, 81
        #     ],
        #     "left": [
        #         0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        #         101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        #     ],
        #     "right": [
        #         0, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119,
        #         120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132
        #     ],
        #     "face": [
        #         23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65,
        #         68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81
        #     ]
        # }

        # self.adj_learn_body = AdjacencyLearn(
        #     n_in_enc=3,
        #     n_hid_enc=256,
        #     edge_types=3,
        #     n_in_dec=256,
        #     n_hid_dec=256,
        #     node_num=len(self.kps_dict['body'])
        # )

        # self.gcn_body = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={'layout': 'body'},
        #     edge_importance_weighting=True,
        #     num_nodes=len(self.kps_dict['body'])
        # )
        # self.gcn_body.fcn = Identity()
        # self.gcn_hand = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={'layout': 'left_hand'},
        #     edge_importance_weighting=True,
        #     num_nodes=len(self.kps_dict['left_hand'])
        # )
        # self.gcn_hand.fcn = Identity()
        # self.gcn_face = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={'layout': 'face'},
        #     edge_importance_weighting=True,
        #     num_nodes=len(self.kps_dict['all_face'])
        # )
        # self.gcn_face.fcn = Identity()

        self.gcn_body = DeGCN(
            in_channels=3,
            num_class=1000,
            graph_args={'layout': 'body'},
            num_point=len(self.kps_dict['body']),
            num_person=1,
            num_stream=1,
        )
        self.gcn_body.fc = Identity()
        self.gcn_hand = DeGCN(
            in_channels=3,
            num_class=1000,
            graph_args={'layout': 'left_hand'},
            num_point=len(self.kps_dict['left_hand']),
            num_person=1,
            num_stream=1,
        )
        self.gcn_hand.fc = Identity()
        self.gcn_face = DeGCN(
            in_channels=3,
            num_class=1000,
            graph_args={'layout': 'face'},
            num_point=len(self.kps_dict['all_face']),
            num_person=1,
            num_stream=1,
        )
        self.gcn_face.fc = Identity()
        self.linear1 = nn.Linear(256 * 4, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.bn = nn.BatchNorm1d(1024, eps=1e-6)
        self.relu = nn.ReLU()

        # self.gcn = STGCN(
        #     in_channels=3,
        #     num_class=1000,
        #     graph_args={},
        #     edge_importance_weighting=True,
        #     num_nodes=121
        # )
        # self.gcn.fcn = Identity()
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LayerNorm(512, eps=1e-6),
        #     nn.ReLU(),
        # )

        # self.conv1d = TemporalConv(
        #     input_size=256,
        #     hidden_size=512,
        #     conv_type=2,
        #     use_bn=True
        # )

        # self.lstm = BiLSTMLayer(
        #     input_size=1024,
        #     hidden_size=1024,
        #     num_layers=2,
        #     dropout=0.1,
        #     bidirectional=True,
        #     rnn_type='LSTM'
        # )

        self.classifier = NormLinear(1024, self.recognition_tokenizer.vocab_size)

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
        # A_body = self.adj_learn_body(kps_body)
        kps_left_hand = kps[:, :, :, self.kps_dict['left_hand'], :]
        # A_left_hand = self.adj_learn(kps_left_hand)
        kps_right_hand = kps[:, :, :, self.kps_dict['right_hand'], :]
        # A_right_hand = self.adj_learn(kps_right_hand)
        kps_face = kps[:, :, :, self.kps_dict['all_face'], :]
        # A_face = self.adj_learn(kps_face)

        f_body = self.gcn_body(kps_body)
        f_left_hand = self.gcn_hand(kps_left_hand)
        f_right_hand = self.gcn_hand(kps_right_hand)
        f_face = self.gcn_face(kps_face)

        f_concat = torch.cat((f_body[0], f_left_hand[0], f_right_hand[0], f_face[0]), dim=1)
        # f_concat = self.gcn(kps[:, :, :,
        #                     self.kps_dict['body'] +
        #                     self.kps_dict['left_hand'] +
        #                     self.kps_dict['right_hand'] +
        #                     self.kps_dict['face'] +
        #                     self.kps_dict['eyes'] +
        #                     self.kps_dict['eyebrows'] +
        #                     self.kps_dict['mouth'] +
        #                     self.kps_dict['nose'],
        #                     :])

        # f_fuse = self.mlp(f_concat.permute(0, 2, 1))

        f_fuse = self.linear1(f_concat.permute(0, 2, 1))
        f_fuse = self.linear2(f_fuse)
        f_fuse = f_fuse.permute(0, 2, 1)
        f_fuse = self.bn(f_fuse)
        f_fuse = self.relu(f_fuse)
        f_fuse = f_fuse.permute(0, 2, 1)

        # f_conv1d = self.conv1d(f_fuse)

        f_fuse = f_fuse.permute(1, 0, 2)  # ntc -> (T, N, C)
        v_len = kps_lgt // 4
        # f_lstm, _ = self.lstm(f_fuse, v_len.cpu())

        return self.classifier(f_fuse), None, v_len

    @override
    def step_forward(self, batch) -> Any:
        kps, y_glosses, y_translation, kps_lgt, y_glosses_lgt, y_translation_lgt, name = batch
        batch_size = len(name)

        outputs = self.forward(kps, kps_lgt)

        if self.trainer.predicting:
            return torch.tensor([]), outputs[1], None, outputs[2], None, name

        loss = (
                1 * self.ctc_loss(outputs[0].log_softmax(-1), y_glosses, outputs[2].cpu(), y_glosses_lgt.cpu()).mean() +0
                # 1 * self.ctc_loss(outputs[1].log_softmax(-1), y_glosses, outputs[2].cpu(), y_glosses_lgt.cpu()).mean() +
                # 25 * self.dist_loss(outputs[0], outputs[1].detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        if "translation" in self.task:
            pass

        return loss, outputs[0], None, outputs[2], None, name


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
