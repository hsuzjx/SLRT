from typing import Any, Union, Sequence

import torch
import wandb
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing_extensions import override

from .modules import STAttentionModule, VisualHead
from ..BaseModel import SLRTBaseModel


class KpsModel(SLRTBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'KpsModel'

    @override
    def _init_network(self, **kwargs):
        self.face_visual_backbone = STAttentionModule(
            num_channel=3,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["face"]),
            st_attention_module_prams=self.hparams.network.st_attention_module_prams,
        )
        self.body_visual_backbone = STAttentionModule(
            num_channel=3,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["body"]),
            st_attention_module_prams=self.hparams.network.st_attention_module_prams,
        )
        self.left_visual_backbone = STAttentionModule(
            num_channel=3,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["left"]),
            st_attention_module_prams=self.hparams.network.st_attention_module_prams,
        )
        self.right_visual_backbone = STAttentionModule(
            num_channel=3,
            max_frame=400,
            num_node=len(self.hparams.kps_idx["right"]),
            st_attention_module_prams=self.hparams.network.st_attention_module_prams,
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

    def forward(self, kps, kps_lgt) -> Any:
        # 获取输入数据的维度
        N, C, T, V = kps.shape

        # 调整输入数据的维度顺序并保持连续性
        kps = kps.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        # backbone forward
        face_backbone_output = self.face_visual_backbone(kps[:, :, :, self.hparams.kps_idx["face"]])
        body_backbone_output = self.body_visual_backbone(kps[:, :, :, self.hparams.kps_idx["body"]])
        left_backbone_output = self.left_visual_backbone(kps[:, :, :, self.hparams.kps_idx["left"]])
        right_backbone_output = self.right_visual_backbone(kps[:, :, :, self.hparams.kps_idx["right"]])

        # 合并数据
        fuse_cat = torch.cat([left_backbone_output, face_backbone_output, right_backbone_output, body_backbone_output],
                             dim=-1)
        left_cat = torch.cat([left_backbone_output, face_backbone_output], dim=-1)
        right_cat = torch.cat([right_backbone_output, face_backbone_output], dim=-1)
        body_cat = torch.cat([body_backbone_output], dim=-1)

        # mask
        mask_lgt = (((kps_lgt - 1) / 2) + 1).long()
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

    def step_forward(self, batch) -> Any:
        kps, y_glosses, y_translation, kps_lgt, y_glosses_lgt, y_translation_lgt, name = batch
        batch_size = len(name)

        outputs = self.forward(kps, kps_lgt)

        if self.trainer.predicting:
            return torch.tensor([]), outputs['ensemble_last_gloss_logits'].permute(1, 0, 2), None, outputs[
                'input_lengths'], None, name

        loss_ctc_list = []
        for k in ['left', 'right', 'fuse', 'body']:
            loss_ctc_list.append(
                self.ctc_loss(
                    log_probs=outputs[f'{k}_gloss_logits'].log_softmax(2).permute(1, 0, 2).to(self.device),
                    targets=y_glosses.to(self.device),
                    input_lengths=outputs['input_lengths'].to(self.device),
                    target_lengths=y_glosses_lgt.to(self.device)
                ) / batch_size
            )
        loss_ctc = loss_ctc_list[0] + loss_ctc_list[1] + loss_ctc_list[2] + loss_ctc_list[3]

        loss_kldiv_list = []
        for student in ['left', 'right', 'fuse', 'body']:
            teacher_prob = outputs['ensemble_last_gloss_logits'].softmax(2)
            teacher_prob = teacher_prob.detach()
            student_log_prob = outputs[f'{student}_gloss_logits'].log_softmax(2)
            loss_kldiv_list.append(
                self.kldiv_loss(
                    input=student_log_prob,
                    target=teacher_prob
                )
            )
        loss_kldiv = loss_kldiv_list[0] + loss_kldiv_list[1] + loss_kldiv_list[2] + loss_kldiv_list[3]

        loss = loss_ctc + loss_kldiv

        if "translation" in self.task:
            pass

        return loss, outputs['ensemble_last_gloss_logits'].permute(1, 0, 2), None, outputs['input_lengths'], None, name

    @override
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        # Initialize the Adam optimizer
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(),
            **self.hparams.optimizer["Adam"]
        )

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            **self.hparams.lr_scheduler["CosineAnnealingLR"]
        )

        # Return the optimizer and learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """

        :return:
        """
        early_stop = EarlyStopping(
            monitor='Val/Loss',
            mode='min',
            **self.hparams.callback["EarlyStopping"]
        )
        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_save_dir,
            monitor='Val/Word-Error-Rate',
            mode='min',
            **self.hparams.callback["ModelCheckpoint"]
        )
        return [early_stop, checkpoint]
