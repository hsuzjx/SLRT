from typing import Any

import torch
from torch import nn

from .SLRBaseModel import SLRBaseModel
from .modules import resnet18, Identity, TemporalConv, NormLinear, BiLSTMLayer, SeqKD
from .utils import Decode


class CorrNet(SLRBaseModel):
    def __init__(self, **kwargs):
        """
        初始化CorrNet模型。
        """
        super().__init__(**kwargs)
        self.model_name = "CorrNet"

        # 定义网络
        self._init_networks()

        # 定义解码器
        self._define_decoder()

        # 定义损失函数
        self._define_loss_function()

    def _init_networks(self):
        """
        初始化模型的各个网络组件。
        """
        self.conv2d = self._init_conv2d()
        self.conv1d = self._init_conv1d()
        self.temporal_model = self._init_bilstm()
        if self.hparams.share_classifier:
            self.classifier = self.conv1d.fc
        else:
            self.classifier = self._init_classifier()

    def forward(self, x, x_lgt) -> Any:
        """
        定义模型的前向传播过程。

        参数:
            x: 输入的视频序列。
            x_lgt: 视频序列的长度。

        返回:
            conv1d_logits: 1D卷积后的输出。
            output_logits: 最终的分类输出。
            feature_lengths: 特征序列的长度。
        """
        batch_size, sequence_length, channels, height, width = x.shape
        reshaped_inputs = x.permute(0, 2, 1, 3, 4)
        convolved = self.conv2d(reshaped_inputs).view(batch_size, sequence_length, -1).permute(0, 2, 1)

        # 通过一维卷积层
        conv1d_output = self.conv1d(convolved, x_lgt)
        visual_features = conv1d_output['visual_feat']
        feature_lengths = conv1d_output['feat_len']
        conv1d_logits = conv1d_output['conv_logits']

        # 通过双向 LSTM 层
        lstm_output = self.temporal_model(visual_features, feature_lengths)

        predictions = lstm_output['predictions']
        # 通过分类器
        output_logits = self.classifier(predictions)

        return conv1d_logits, output_logits, feature_lengths

    def step_forward(self, batch):
        """
        执行单个前向传播步骤并计算损失。

        参数:
            batch: 一个包含输入数据和标签的批次。

        返回:
            如果在训练模式下，返回计算得到的损失。
            如果不在训练模式下，返回损失、解码后的输出和相关信息。
        """
        x, x_lgt, y, y_lgt, info = batch
        # 模型正向传播
        conv1d_hat, y_hat, y_hat_lgt = self(x, x_lgt)

        conv1d_hat_softmax = conv1d_hat.log_softmax(-1)
        y_hat_softmax = y_hat.log_softmax(-1)

        # y = y.cpu().int()
        # y_hat_lgt = y_hat_lgt.cpu().int()
        # y_lgt = y_lgt.cpu().int()

        # 计算损失
        loss = self.hparams.loss_weights[0] * self.ctc_loss(conv1d_hat_softmax, y, y_hat_lgt, y_lgt).mean() + \
               self.hparams.loss_weights[1] * self.ctc_loss(y_hat_softmax, y, y_hat_lgt, y_lgt).mean() + \
               self.hparams.loss_weights[2] * self.dist_loss(conv1d_hat, y_hat.detach(), use_blank=False)

        # 检查是否有 NaN 值
        if torch.isnan(loss):
            print('\nWARNING:Detected NaN in loss.')

        if self.training:
            return loss
        else:
            decoded = self.decoder.decode(y_hat, y_hat_lgt, batch_first=False, probs=False)
            assert len(info) == len(decoded)
            return loss, decoded, info

    def _define_loss_function(self):
        """
        定义损失函数。
        """
        self.ctc_loss = nn.CTCLoss(reduction='none', zero_infinity=False)
        self.dist_loss = SeqKD(T=8)

    def _define_decoder(self):
        """
        初始化解码器。
        """
        self.decoder = Decode(
            gloss_dict=self.hparams.gloss_dict,
            num_classes=self.hparams.num_classes,
            search_mode='beam'
        )

    def _init_conv2d(self):
        """
        初始化2D卷积层。

        返回:
            使用ResNet-18作为2D卷积层，去除其全连接层。
        """
        conv2d = resnet18()
        conv2d.fc = Identity()
        return conv2d

    def _init_conv1d(self):
        """
        初始化1D卷积层。

        返回:
            包含全连接层的1D卷积层。
        """
        conv1d = TemporalConv(
            input_size=512,
            hidden_size=self.hparams.hidden_size,
            conv_type=self.hparams.conv_type,
            use_bn=self.hparams.use_bn,
            num_classes=self.hparams.num_classes
        )
        conv1d.fc = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)
        return conv1d

    def _init_bilstm(self):
        """
        初始化双向LSTM层。

        返回:
            一个双向LSTM层，用于序列建模。
        """
        return BiLSTMLayer(
            rnn_type='LSTM',
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=2,
            bidirectional=True
        )

    def _init_classifier(self):
        """
        初始化分类器。

        返回:
            一个线性分类器，用于将LSTM的输出映射到类别上。
        """
        return NormLinear(self.hparams.hidden_size, self.hparams.num_classes)
