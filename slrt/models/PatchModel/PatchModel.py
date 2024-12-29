from typing import Any, Union, Sequence, Tuple

import torch
import torch.nn.functional as F
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from typing_extensions import override

from .modules import ResNet18, TemporalConv, BiLSTMLayer, SeqKD
from ..BaseModel import SLRTBaseModel
from ..XModel.modules import SeqKD


class PatchModel(SLRTBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PatchModel"

    @override
    def _init_network(self, **kwargs):
        self.visual_local_layer = ResNet18(
            **self.hparams.network["ResNet18"]
        )
        self.pool_layer = None

        # self.kps_linear_layer = nn.Linear()

        self.temporal_conv_layer = TemporalConv(
            **self.hparams.network["conv1d"]
        )

        self.visual_gat = None
        self.temporal_gat = None

        self.visual_conv_layer = TemporalConv(
            **self.hparams.network["conv1d"]
        )

        self.bilstm = BiLSTMLayer(
            **self.hparams.network["BiLSTM"]
        )
        self.classifier = NormLinear(1024, self.recognition_tokenizer.vocab_size)

        if not self.hparams.network['share_classifier']:
            self.classifier_vconv_fc = NormLinear(1024, self.recognition_tokenizer.vocab_size)

    def forward(self, patchs: torch.Tensor, patch_lengths: torch.Tensor,
                kps: torch.Tensor) -> Any:
        N, T, C, V, H, W = patchs.shape
        # N, _, T, V = kps.shape

        x = patchs.permute(0, 3, 2, 1, 4, 5).contiguous().view(N * V, C, T, H, W)

        # (N * V, C, T, H, W) -> (N * V, C, T, H', W')
        x = self.visual_local_layer(x)

        # (N * V, C, T, H', W') -> (N * V, T, f)
        # ..... mean pooling
        x = self.pool_layer(x)

        x = torch.cat([x, kps.view(N * T, V, 3)], dim=2)

        # (N * V, T, f) -> (N * V, f, T) -> (N * V, f, T')
        x, x_lengths = self.temporal_conv_layer(x.permute(0, 2, 1).contiguous(), patch_lengths)

        # (N * V, f, T') -> (N, V, f, T')
        x = x.view(N, V, -1, x_lengths)

        # (N, V, f, T') -> (N, T', V, f) -> (N, T', V, f_v)
        visual_features = self.visual_gat(x.permute(0, 3, 1, 2).contiguous())

        # (N, V, f, T') -> (N, V, T', f) -> (N, V, T', f_t) -> (N, T', V, f_t))
        temporal_features = self.temporal_gat(x.permute(0, 1, 3, 2).contiguous()).permute(0, 2, 1, 3).contiguous()

        # concat(visual_features, temporal_features) -> (N, T', V, 2f)
        gat_out_features = torch.cat([visual_features, temporal_features], dim=3)

        # (N, T', V, 2f) -> (N, T', f, V) -> (N * T', f, V)) -> (N * T', f_bilstm_in))
        bilstm_in = self.visual_conv_layer(gat_out_features.permute(0, 1, 3, 2).contiguous().view(N * x_lengths, -1, V))

        # (N * T', f_bilstm_in) -> (N, T', f_bilstm_in) -> (T', N, f_bilstm_in)
        bilstm_in = bilstm_in.view(N, x_lengths, -1).permute(1, 0, 2).contiguous()
        # (N, T', V, 2f) -> (N, T', V, 2f)
        predictions, _ = self.bilstm(bilstm_in, x_lengths)

        logits = self.classifier(predictions)

        if self.hparams.network['share_classifier']:
            vconv_logits = self.classifier(bilstm_in.view(N, x_lengths, -1))
        else:
            vconv_logits = self.classifier_vconv_fc(bilstm_in.view(N, x_lengths, -1))

        return vconv_logits, logits, x_lengths

    @override
    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            **self.hparams.loss_fn['CTCLoss']
        )
        self.dist_loss = SeqKD(
            **self.hparams.loss_fn['SeqKD']
        )

    def step_forward(
            self,
            batch: Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any
            ]
    ) -> Tuple[torch.Tensor, Any, Any, Any, Any, Any]:
        patches, kps, glosses, words, patches_length, kps_length, gloss_lengths, words_lengths, info = batch
        vconv_hat, y_hat, lgt = self(patches, patches_length, kps)

        if self.trainer.predicting:
            return torch.tensor([]), y_hat, None, lgt, None, info

        loss = (
                1 * self.ctc_loss(vconv_hat.log_softmax(-1), glosses, lgt.cpu(), gloss_lengths.cpu()).mean() +
                1 * self.ctc_loss(y_hat.log_softmax(-1), glosses, lgt.cpu(), gloss_lengths.cpu()).mean() +
                25 * self.dist_loss(vconv_hat, y_hat.detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, y_hat, None, lgt, None, info

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
