from typing import Any

import torch

from slr.models.SLRBaseModel import SLRBaseModel
from slr.models.modules.mska.DSTA import DSTA
from slr.models.modules.mska.Visualhead import VisualHead


class MSKA(SLRBaseModel):
    """
    MSKA model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.name = 'MSKA'

        self._init_network(**kwargs)

        self._define_loss_function()

        self.tokenizer = self.hparams.probs_decoder.tokenizer

    def __init_network(self, **kwargs):
        self.visual_backbone = None
        self.rgb_visual_head = None
        self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'], num_channel=3, args=self.hparams)
        self.fuse_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
        self.body_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['body_visual_head'])
        self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
        self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])

    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

    def forward(self, kps, ) -> Any:
        fuse, left_output, right_output, body = self.visual_backbone_keypoint(kps)
        body_head = self.body_visual_head(
            x=body,
            mask=src_input['mask'].cuda(),
            valid_len_in=src_input['new_src_lengths'].cuda())
        fuse_head = self.fuse_visual_head(
            x=fuse,
            mask=src_input['mask'].cuda(),
            valid_len_in=src_input['new_src_lengths'].cuda())
        left_head = self.left_visual_head(
            x=left_output,
            mask=src_input['mask'].cuda(),
            valid_len_in=src_input['new_src_lengths'].cuda())
        right_head = self.right_visual_head(
            x=right_output,
            mask=src_input['mask'].cuda(),
            valid_len_in=src_input['new_src_lengths'].cuda())
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

        head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(
            2)
        head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
        self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble', 'gloss_feature')
        head_outputs['gloss_feature'] = fuse_head[self.cfg['gloss_feature_ensemble']]

        ############
        outputs = {**head_outputs,
                   'input_lengths': src_input['new_src_lengths']}

        for k in ['left', 'right', 'fuse', 'body']:
            outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                gloss_labels=src_input['gloss_input']['gloss_labels'].cuda(),
                gloss_lengths=src_input['gloss_input']['gls_lengths'].cuda(),
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['new_src_lengths'].cuda())
        outputs['recognition_loss'] = outputs['recognition_loss_left'] + outputs['recognition_loss_right'] + \
                                      outputs['recognition_loss_fuse'] + outputs['recognition_loss_body']
        if 'cross_distillation' in self.cfg:
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for student in ['left', 'right', 'fuse', 'body']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        return outputs

    def step_forward(self, batch) -> Any:
        outputs = self(batch)
