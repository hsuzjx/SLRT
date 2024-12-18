from typing import Any

import torch
from typing_extensions import override

from slrt.models.BaseModel import SLRBaseModel
from slrt.models.STKA.layers import STAttentionModule, VisualHead


class STKA(SLRBaseModel):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.name = 'STKA'

        self.__init_network(**kwargs)
        self._define_loss_function()
        self.glosses_decoder = self.hparams.probs_decoder

    def __init_network(self, **kwargs):
        self.backbone = STAttentionModule(
            num_channel=3,
            max_frame=400,
            num_node=133,
            st_attention_module_prams=self.hparams.network.st_attention_module_prams
        )

        num_classes = len(self.hparams.probs_decoder.tokenizer)
        self.head = VisualHead(
            cls_num=num_classes,
            **self.hparams.network.head_cfg
        )

    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

    def forward(self, kps, kps_lgt) -> Any:
        # 获取输入数据的维度
        N, C, T, V = kps.shape
        # 调整输入数据的维度顺序并保持连续性
        kps = kps.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        backbone_output = self.backbone(kps)

        # mask
        mask_lgt = (((kps_lgt - 1) / 2) + 1).long()
        mask_lgt = (((mask_lgt - 1) / 2) + 1).long()
        max_len = torch.max(mask_lgt)
        mask = torch.zeros(mask_lgt.shape[0], 1, max_len)
        for i in range(mask_lgt.shape[0]):
            mask[i, :, :mask_lgt[i]] = 1
        mask = mask.to(torch.bool)

        logits = self.head(x=backbone_output, mask=mask.to(self.device))
        return logits, mask_lgt

    def step_forward(self, batch) -> Any:
        x, y_glosses, y_translation, x_lgt, y_glosses_lgt, y_translation_lgt, name = batch
        batch_size = len(name)

        logits, input_lengths = self.forward(x, x_lgt)

        loss_ctc = self.ctc_loss(
            log_probs=logits.log_softmax(2).permute(1, 0, 2).to(self.device),
            targets=y_glosses.to(self.device),
            input_lengths=input_lengths.to(self.device),
            target_lengths=y_glosses_lgt.to(self.device)
        ) / batch_size

        return loss_ctc, logits.log().permute(1, 0, 2), input_lengths, name

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
