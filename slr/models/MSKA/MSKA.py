from typing import Any

import torch

from slr.models.BaseModel.SLRBaseModel import SLRBaseModel
from slr.models.MSKA.modules.DSTA import DSTA
from slr.models.MSKA.modules.Visualhead import VisualHead


class MSKA(SLRBaseModel):
    """
    MSKA model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.name = 'MSKA'

        self.__init_network(**kwargs)

        self._define_loss_function()

        self.probs_decoder = self.hparams.probs_decoder

        # self.gloss_tokenizer = self.hparams.probs_decoder.tokenizer

    def __init_network(self, **kwargs):
        self.visual_backbone = None
        self.rgb_visual_head = None
        self.visual_backbone_keypoint = DSTA(net_prams=self.hparams.net_prams, num_channel=3)

        cfg = self.hparams.head_cfg

        self.fuse_visual_head = VisualHead(
            cls_num=len(self.hparams.probs_decoder.tokenizer),
            **cfg['fuse_visual_head']
        )
        self.body_visual_head = VisualHead(
            cls_num=len(self.hparams.probs_decoder.tokenizer),
            **cfg['body_visual_head']
        )
        self.left_visual_head = VisualHead(
            cls_num=len(self.hparams.probs_decoder.tokenizer),
            **cfg['left_visual_head']
        )
        self.right_visual_head = VisualHead(
            cls_num=len(self.hparams.probs_decoder.tokenizer),
            **cfg['right_visual_head']
        )

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
        fuse, left_output, right_output, body = self.visual_backbone_keypoint(
            kps=kps,
            **self.hparams.kps_parts_idx
        )

        mask_lgt = (((kps_lgt - 1) / 2) + 1).long()
        mask_lgt = (((mask_lgt - 1) / 2) + 1).long()

        max_len = max(mask_lgt)
        mask = torch.zeros(mask_lgt.shape[0], 1, max_len)
        for i in range(mask_lgt.shape[0]):
            mask[i, :, :mask_lgt[i]] = 1
        mask = mask.to(torch.bool)

        body_head = self.body_visual_head(
            x=body,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )
        fuse_head = self.fuse_visual_head(
            x=fuse,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )
        left_head = self.left_visual_head(
            x=left_output,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )
        right_head = self.right_visual_head(
            x=right_output,
            mask=mask.to(self.device),
            valid_len_in=mask_lgt.to(self.device)
        )

        head_outputs = {
            'ensemble_last_gloss_logits': (
                    left_head['gloss_probabilities'] + right_head['gloss_probabilities'] +
                    body_head['gloss_probabilities'] + fuse_head['gloss_probabilities']
            ).log(),
            'fuse': fuse,
            'fuse_gloss_logits': fuse_head['gloss_logits'],
            'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],
            'body_gloss_logits': body_head['gloss_logits'],
            'body_gloss_probabilities_log': body_head['gloss_probabilities_log'],
            'left_gloss_logits': left_head['gloss_logits'],
            'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],
            'right_gloss_logits': right_head['gloss_logits'],
            'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
        }

        head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs[
            'ensemble_last_gloss_logits'
        ].log_softmax(2)

        head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
        # self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble', 'gloss_feature')
        # head_outputs['gloss_feature'] = fuse_head[self.cfg['gloss_feature_ensemble']]

        ############
        outputs = {**head_outputs,
                   'input_lengths': mask_lgt}

        # for k in ['left', 'right', 'fuse', 'body']:
        #     outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
        #         gloss_labels=src_input['gloss_input']['gloss_labels'].cuda(),
        #         gloss_lengths=src_input['gloss_input']['gls_lengths'].cuda(),
        #         gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
        #         input_lengths=src_input['new_src_lengths'].cuda())
        # outputs['recognition_loss'] = outputs['recognition_loss_left'] + outputs['recognition_loss_right'] + \
        #                               outputs['recognition_loss_fuse'] + outputs['recognition_loss_body']

        # if 'cross_distillation' in self.cfg:
        #     loss_func = torch.nn.KLDivLoss(reduction="batchmean")
        #     for student in ['left', 'right', 'fuse', 'body']:
        #         teacher_prob = outputs['ensemble_last_gloss_probabilities']
        #         teacher_prob = teacher_prob.detach()
        #         student_log_prob = outputs[f'{student}_gloss_probabilities_log']
        #         outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
        #         outputs['recognition_loss'] += outputs[f'{student}_distill_loss']

        return outputs

    def step_forward(self, batch) -> Any:
        # x = batch['kps']
        # y_glosses = batch['glosses']
        # y_translation = batch['translation']
        # x_lgt = batch['kps_length']
        # y_glosses_lgt = batch['glosses_length']
        # y_translation_lgt = batch['translation_length']
        # name = batch['name']

        x, y_glosses, y_translation, x_lgt, y_glosses_lgt, y_translation_lgt, name = batch

        outputs = self.forward(x, x_lgt)

        loss_ctc_list = []
        for k in ['left', 'right', 'fuse', 'body']:
            loss_ctc_list.append(
                self.ctc_loss(
                    log_probs=outputs[f'{k}_gloss_probabilities_log'].permute(1, 0, 2).to(self.device),
                    targets=y_glosses.to(self.device),
                    input_lengths=outputs['input_lengths'].to(self.device),
                    target_lengths=y_glosses_lgt.to(self.device)
                )
            )
        loss_ctc = loss_ctc_list[0] + loss_ctc_list[1] + loss_ctc_list[2] + loss_ctc_list[3]

        loss_kldiv_list = []
        for student in ['left', 'right', 'fuse', 'body']:
            teacher_prob = outputs['ensemble_last_gloss_probabilities']
            teacher_prob = teacher_prob.detach()
            student_log_prob = outputs[f'{student}_gloss_probabilities_log']
            loss_kldiv_list.append(
                self.kldiv_loss(
                    input=student_log_prob,
                    target=teacher_prob
                )
            )
        loss_kldiv = loss_kldiv_list[0] + loss_kldiv_list[1] + loss_kldiv_list[2] + loss_kldiv_list[3]

        loss = loss_ctc + loss_kldiv

        return loss, outputs['ensemble_last_gloss_logits'].permute(1, 0, 2), outputs['input_lengths'], name
