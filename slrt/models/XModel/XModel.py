import shutil
from typing import Any, Tuple, Union, Sequence

import torch
import torch.nn.functional as F
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from typing_extensions import override

from .modules import *
from ..BaseModel import SLRTBaseModel


class XModel(SLRTBaseModel):
    """
    XModel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "XModel"
        shutil.copy("models/XModel/XModel.py", self.hparams.save_dir)

    @override
    def _init_network(self, **kwargs):
        self.visual_backbone = S3D_backbone(
            in_channel=3,
            **self.hparams.network['s3d'],
            cfg_pyramid=self.hparams.network['pyramid']
        )

        self.temporal_module = BiLSTMLayer(
            rnn_type='LSTM',
            input_size=832,
            hidden_size=832,
            num_layers=2,
            bidirectional=True
        )

        self.classifier = NormLinear(832, self.recognition_tokenizer.vocab_size)
        if not self.hparams.network['share_classifier']:
            self.classifier_conv_fc = NormLinear(832, self.recognition_tokenizer.vocab_size)

    def forward(
            self,
            videos: torch.Tensor,
            video_lengths: torch.Tensor,
            glosses: torch.Tensor = None,
            gloss_lengths: torch.Tensor = None,
            words: torch.Tensor = None,
            words_lengths: torch.Tensor = None
    ) -> Any:
        N, T, C, H, W = videos.shape

        videos = videos / 127.5 - 1

        # (N,T,C,H,W) -> (N,C,T,H,W)
        x = videos.permute(0, 2, 1, 3, 4).contiguous().view(N, C, T, H, W)

        # (N,C,T,H,W) -> out
        out = self.visual_backbone(x, sgn_lengths=video_lengths)
        sgn_feature, sgn_mask, valid_len_out, fea_lst = out['sgn_feature'], out['sgn_mask'], out['valid_len_out'], out[
            'fea_lst']

        # (N,T,features) -> (T,N,features)
        visual_features = sgn_feature.permute(1, 0, 2)

        # (T',N,features') -> (T',N,features'')
        predictions, _ = self.temporal_module(visual_features, valid_len_out[-1].cpu())

        # Pass through the classifier
        # (T',N,features'') -> (T',N,logits)
        logits = self.classifier(predictions)

        if visual_features.shape[0] != predictions.shape[0]:
            print(f"\n{visual_features.shape[0]} {predictions.shape[0]}")

        if self.hparams.network['share_classifier']:
            # (T',N,features') -> (T',N,logits_conv)
            conv_logits = self.classifier(visual_features[:max(valid_len_out[-1]), :, :])
        else:
            # (T',N,features') -> (T',N,logits_conv)
            conv_logits = self.classifier_conv_fc(visual_features)

        return {
            "logits": logits,
            "logits_length": valid_len_out[-1],
            "conv_logits": conv_logits,
            "vlo": valid_len_out
        }

    def step_forward(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> Tuple[torch.Tensor, Any, Any, Any, Any, Any]:
        """
        Performs a forward pass and computes the loss for a given batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]): A tuple containing the input data, input lengths, target data, target lengths, and additional information.

        Returns:
            Tuple[torch.Tensor, Any, Any, Any, Any, Any]: Loss value, softmax predictions, predicted lengths, and additional information.
        """
        x, y, _, x_lgt, y_lgt, _, info = batch
        out = self(x, x_lgt)
        y_hat, y_hat_lgt, conv_hat = out['logits'], out['logits_length'], out['conv_logits']

        if self.trainer.predicting:
            return torch.tensor([]), y_hat, None, y_hat_lgt, None, info

        loss = (
                1 * self.ctc_loss(conv_hat.log_softmax(-1), y, y_hat_lgt.cpu(), y_lgt.cpu()).mean() +
                1 * self.ctc_loss(y_hat.log_softmax(-1), y, y_hat_lgt.cpu(), y_lgt.cpu()).mean() +
                25 * self.dist_loss(conv_hat, y_hat.detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, y_hat, None, y_hat_lgt, None, info

    @override
    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            **self.hparams.loss_fn['CTCLoss']
        )
        self.dist_loss = SeqKD(
            **self.hparams.loss_fn['SeqKD']
        )

    @override
    def configure_optimizers(self):
        # Adam
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(),
            **self.hparams.optimizer['Adam']
        )

        # MultiStepLR
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            **self.hparams.lr_scheduler['MultiStepLR']
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    @override
    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """

        :return:
        """
        early_stop = EarlyStopping(
            monitor='Val/Loss',
            mode='min',
            **self.hparams.callback['EarlyStopping']
        )
        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_save_dir,
            monitor='Val/Word-Error-Rate',
            mode='min',
            **self.hparams.callback['ModelCheckpoint']
        )
        return [early_stop, checkpoint]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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
