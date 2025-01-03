from typing import Any

import torch
from typing_extensions import override

from slrt.models.BaseModel.SLRTBaseModel import SLRTBaseModel
from slrt.models.MSKA.modules.BiLSTM import BiLSTMLayer
from slrt.models.MSKA.modules.DSTA import STAttentionModule
from slrt.models.MSKA.modules.Visualhead import VisualHead
# from slrt.models.MSKA.modules.translation import TranslationNetwork
from slrt.models.MSKA.modules.vl_mapper import VLMapper


class MSKA(SLRTBaseModel):
    """
    MSKA model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.name = 'MSKA'

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

        self.fuse_bilstm_layer = BiLSTMLayer(**self.hparams.network["fuse_BiLSTM"])
        self.body_bilstm_layer = BiLSTMLayer(**self.hparams.network["body_BiLSTM"])
        self.left_bilstm_layer = BiLSTMLayer(**self.hparams.network["left_BiLSTM"])
        self.right_bilstm_layer = BiLSTMLayer(**self.hparams.network["right_BiLSTM"])

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

        # if "translation" in self.task:
        #     self.translation_network = TranslationNetwork(cfg=self.hparams['TranslationNetwork'])
        #
        #     if self.hparams['VLMapper'].get('type', 'projection') == 'projection':
        #         if 'in_features' in self.hparams['VLMapper']:
        #             in_features = self.hparams['VLMapper'].pop('in_features')
        #         else:
        #             in_features = 512
        #     else:
        #         in_features = len(self.recognition_tokenizer)
        #
        #     self.vl_mapper = VLMapper(
        #         cfg=self.hparams['VLMapper'],
        #         in_features=in_features,
        #         out_features=self.translation_network.input_dim,
        #         gloss_id2str=self.recognition_tokenizer.ids_to_vocab,
        #         gls2embed=getattr(self.translation_network, 'gls2embed', None)
        #     )

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

        valid_gloss_length = kps_lgt // 4
        fuse_cat = fuse_cat.permute(1, 0, 2)
        body_cat = body_cat.permute(1, 0, 2)
        left_cat = left_cat.permute(1, 0, 2)
        right_cat = right_cat.permute(1, 0, 2)
        fuse_cat, _ = self.fuse_bilstm_layer(fuse_cat, valid_gloss_length.cpu())
        body_cat, _ = self.body_bilstm_layer(body_cat, valid_gloss_length.cpu())
        left_cat, _ = self.left_bilstm_layer(left_cat, valid_gloss_length.cpu())
        right_cat, _ = self.right_bilstm_layer(right_cat, valid_gloss_length.cpu())
        fuse_cat = fuse_cat.permute(1, 0, 2).contiguous()
        body_cat = body_cat.permute(1, 0, 2).contiguous()
        left_cat = left_cat.permute(1, 0, 2).contiguous()
        right_cat = right_cat.permute(1, 0, 2).contiguous()

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

        # if "translation" in self.task:
        #     mapped_feature = self.vl_mapper(visual_outputs={"gloss_feature": fuse_head_output["gloss_probabilities"]})
        #     translation_inputs = {
        #         **src_input['translation_inputs'],
        #         'input_feature': mapped_feature,
        #         'input_lengths': mask_lgt}
        #     translation_outputs = self.translation_network(**translation_inputs)
        #     model_outputs = {**translation_outputs, **recognition_outputs}
        #     model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']  # for latter use of decoding
        #     model_outputs['total_loss'] = model_outputs['recognition_loss'] + model_outputs['translation_loss']
        #
        #     return {
        #         'gloss_logits': fuse_head_output,
        #         'input_lengths': mask_lgt
        #     }

        return {
            'fuse_gloss_logits': fuse_head_output,
            'left_gloss_logits': left_head_output,
            'right_gloss_logits': right_head_output,
            'body_gloss_logits': body_head_output,
            'ensemble_last_gloss_logits': (left_head_output.softmax(dim=2) + right_head_output.softmax(dim=2) +
                                           body_head_output.softmax(dim=2) + fuse_head_output.softmax(dim=2)).log(),
            'input_lengths': mask_lgt
        }

        # return {
        #     'fuse_gloss_logits': fuse_head_output["gloss_logits"],
        #     'left_gloss_logits': left_head_output["gloss_logits"],
        #     'right_gloss_logits': right_head_output["gloss_logits"],
        #     'body_gloss_logits': body_head_output["gloss_logits"],
        #     'ensemble_last_gloss_logits': (
        #             left_head_output['gloss_probabilities'] + right_head_output['gloss_probabilities'] +
        #             body_head_output['gloss_probabilities'] + fuse_head_output['gloss_probabilities']
        #     ).log(),
        #     'input_lengths': mask_lgt
        # }

    def step_forward(self, batch) -> Any:
        # x = batch['kps']
        # y_glosses = batch['glosses']
        # y_translation = batch['translation']
        # x_lgt = batch['kps_length']
        # y_glosses_lgt = batch['glosses_length']
        # y_translation_lgt = batch['translation_length']
        # name = batch['name']

        x, y_glosses, y_translation, x_lgt, y_glosses_lgt, y_translation_lgt, name = batch
        batch_size = len(name)

        outputs = self.forward(x, x_lgt)

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
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
            betas=self.hparams.optimizer.betas,
        )

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.lr_scheduler.get("T_max", 20),
        )

        # Return the optimizer and learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
