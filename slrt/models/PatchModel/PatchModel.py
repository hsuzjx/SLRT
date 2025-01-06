import shutil
from typing import Any, Union, Sequence, Tuple

import torch
import torch.nn.functional as F
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from typing_extensions import override

from .modules import *
from ..BaseModel import SLRTBaseModel


class PatchModel(SLRTBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PatchModel"
        shutil.copy('models/PatchModel/PatchModel.py', self.hparams.save_dir)

    @override
    def _init_network(self, **kwargs):
        self.resnet18_l2 = ResNet18(
            **self.hparams.network["ResNet18"]
        )
        self.vhead = VHead(k=1, s=1)

        config = [
            [128, 128, 32, 7, 2],
            [128, 128, 32, 3, 1],
            [128, 256, 64, 3, 1],
            [256, 256, 64, 3, 1],
            [256, 512, 128, 3, 2],
            [512, 512, 128, 3, 1],
            [512, 512, 128, 3, 1],
        ]
        self.face_visual_backbone = STAttentionModule(
            num_channel=128,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["face"]),
            st_attention_module_prams=config
        )
        self.body_visual_backbone = STAttentionModule(
            num_channel=128,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["body"]),
            st_attention_module_prams=config
        )
        self.left_visual_backbone = STAttentionModule(
            num_channel=128,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["left"]),
            st_attention_module_prams=config
        )
        self.right_visual_backbone = STAttentionModule(
            num_channel=128,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["right"]),
            st_attention_module_prams=config
        )

        num_classes = len(self.recognition_tokenizer)
        self.fuse_visual_head = VisualHead(
            cls_num=num_classes,
            **self.hparams.network.head_cfg['fuse_visual_head']
        )
        self.body_visual_head = VisualHead(
            cls_num=num_classes,
            **self.hparams.network.head_cfg['body_visual_head']
        )
        self.left_visual_head = VisualHead(
            cls_num=num_classes,
            **self.hparams.network.head_cfg['left_visual_head']
        )
        self.right_visual_head = VisualHead(
            cls_num=num_classes,
            **self.hparams.network.head_cfg['right_visual_head']
        )

    def forward(self, patchs: torch.Tensor, patch_lengths: torch.Tensor,
                kps: torch.Tensor) -> Any:
        PN, PT, PV, PC, PH, PW = patchs.shape
        KN, KT, KV, KC = kps.shape

        patchs = patchs.permute(0, 2, 3, 1, 4, 5).contiguous().view(PN * PV, PC, PT, PH, PW)
        kps = kps.permute(0, 3, 1, 2).contiguous().view(KN, KC, KT, KV)

        patchs = self.resnet18_l2(patchs)
        patchs = self.vhead(patchs)
        patchs = patchs.view(PN, PV, PT, -1).permute(0, 3, 2, 1).contiguous().view(PN, 128, PT, PV)

        # backbone forward
        face_backbone_output = self.face_visual_backbone(patchs[:, :, :, self.hparams.kps_idx["face"]])
        body_backbone_output = self.body_visual_backbone(patchs[:, :, :, self.hparams.kps_idx["body"]])
        left_backbone_output = self.left_visual_backbone(patchs[:, :, :, self.hparams.kps_idx["left"]])
        right_backbone_output = self.right_visual_backbone(patchs[:, :, :, self.hparams.kps_idx["right"]])

        # 合并数据
        fuse_cat = torch.cat([left_backbone_output, face_backbone_output, right_backbone_output, body_backbone_output],
                             dim=-1)
        left_cat = torch.cat([left_backbone_output, face_backbone_output], dim=-1)
        right_cat = torch.cat([right_backbone_output, face_backbone_output], dim=-1)
        body_cat = torch.cat([body_backbone_output], dim=-1)

        # mask
        mask_lgt = (((patch_lengths - 1) / 2) + 1).long()
        mask_lgt = (((mask_lgt - 1) / 2) + 1).long()
        max_len = torch.max(mask_lgt)
        mask = torch.zeros(mask_lgt.shape[0], 1, max_len)
        for i in range(mask_lgt.shape[0]):
            mask[i, :, :mask_lgt[i]] = 1
        mask = mask.to(torch.bool)

        # head forward
        fuse_head_output = self.fuse_visual_head(
            x=fuse_cat,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )
        left_head_output = self.left_visual_head(
            x=left_cat,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )
        right_head_output = self.right_visual_head(
            x=right_cat,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )
        body_head_output = self.body_visual_head(
            x=body_cat,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )

        return {
            'fuse_gloss_logits': fuse_head_output,
            'left_gloss_logits': left_head_output,
            'right_gloss_logits': right_head_output,
            'body_gloss_logits': body_head_output,
            'ensemble_last_gloss_logits': (left_head_output.softmax(dim=2) + right_head_output.softmax(dim=2) +
                                           body_head_output.softmax(dim=2) + fuse_head_output.softmax(dim=2)).log(),
            'input_lengths': mask_lgt
        }

    @override
    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

        self.kldiv_loss = torch.nn.KLDivLoss(
            reduction="batchmean"
        )

    def step_forward(
            self,
            batch: Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any
            ]
    ) -> Tuple[torch.Tensor, Any, Any, Any, Any, Any]:
        patches, kps, glosses, words, patches_length, kps_length, gloss_lengths, words_lengths, info = batch
        batch_size = len(info)

        out = self(patches, patches_length, kps)

        lgt = out['input_lengths']

        if self.trainer.predicting:
            return torch.tensor([]), out['ensemble_last_gloss_logits'].permute(1, 0, 2), None, lgt, None, info

        loss_ctc_list = []
        for k in ['left', 'right', 'fuse', 'body']:
            loss_ctc_list.append(
                self.ctc_loss(
                    log_probs=out[f'{k}_gloss_logits'].log_softmax(2).permute(1, 0, 2).to(self.device),
                    targets=glosses.to(self.device),
                    input_lengths=lgt.to(self.device),
                    target_lengths=gloss_lengths.to(self.device)
                ) / batch_size
            )
        loss_ctc = loss_ctc_list[0] + loss_ctc_list[1] + loss_ctc_list[2] + loss_ctc_list[3]

        loss_kldiv_list = []
        for student in ['left', 'right', 'fuse', 'body']:
            teacher_prob = out['ensemble_last_gloss_logits'].softmax(2)
            teacher_prob = teacher_prob.detach()
            student_log_prob = out[f'{student}_gloss_logits'].log_softmax(2)
            loss_kldiv_list.append(
                self.kldiv_loss(
                    input=student_log_prob,
                    target=teacher_prob
                )
            )
        loss_kldiv = loss_kldiv_list[0] + loss_kldiv_list[1] + loss_kldiv_list[2] + loss_kldiv_list[3]

        loss = loss_ctc + loss_kldiv

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, out['ensemble_last_gloss_logits'].permute(1, 0, 2), None, lgt, None, info

    @override
    def configure_optimizers(self):
        # Adam
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(),
            **self.hparams.optimizer['Adam']
        )

        # MultiStepLR
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            **self.hparams.lr_scheduler['CosineAnnealingLR']
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


class VHead(nn.Module):
    def __init__(self, k, s=1):
        super(VHead, self).__init__()

        self.avgpool = nn.AvgPool2d(k, stride=s)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        # 调整张量形状为(batch_size*T, C, H, W)
        x = x.view((-1,) + x.size()[2:])

        # 全局平均池化
        x = self.avgpool(x)
        # 将张量展平
        x = x.view(x.size(0), -1)

        return x
