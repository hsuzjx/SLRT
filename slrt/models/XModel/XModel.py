from typing import Any, Tuple, Union, Sequence

import torch
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing_extensions import override

from slrt.models.BaseModel import SLRTBaseModel
from slrt.models.XModel.modules import ResNet34
from slrt.models.XModel.modules.TemproalModules import BiLSTMLayer


class XModel(SLRTBaseModel):
    """
    XModel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "XModel"

    @override
    def _init_network(self, **kwargs):
        self.visual_backbone = ResNet34(
            num_classes=self.recognition_tokenizer.vocab_size,
            **self.hparams.network['ResNet34']
        )

        # self.visual_head = None

        self.conv1d =None


        self.temporal_module = BiLSTMLayer(
            **self.hparams.network['BiLSTM']
        )

    @override
    def _define_loss_function(self):
        self.ctc_loss = torch.nn.CTCLoss(
            **self.hparams.loss_fn['CTCLoss']
        )
        self.kl_loss = torch.nn.KLDivLoss(
            **self.hparams.loss_fn['KLDivLoss']
        )

    @override
    def configure_optimizers(self):
        # Adam
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(),
            **self.hparams.optimizer['Adam']
        )

        # CosineAnnealingLR
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
        x = self.visual_backbone(videos)
        x = x.permute(0, 2, 1)

    def step_forward(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> Tuple[torch.Tensor, Any, Any, Any, Any, Any]:
        """
        Performs a forward pass and computes the loss for a given batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]): A tuple containing the input data, input lengths, target data, target lengths, and additional information.

        Returns:
            Tuple[torch.Tensor, Any, Any, Any]: Loss value, softmax predictions, predicted lengths, and additional information.
        """
        x, y, _, x_lgt, y_lgt, _, info = batch
        conv1d_hat, y_hat, y_hat_lgt = self(x, x_lgt)

        if self.trainer.predicting:
            return torch.tensor([]), y_hat, None, y_hat_lgt, None, info

        loss = (
                1 * self.ctc_loss(conv1d_hat.log_softmax(-1), y, y_hat_lgt, y_lgt).mean() +
                1 * self.ctc_loss(y_hat.log_softmax(-1), y, y_hat_lgt, y_lgt).mean() +
                25 * self.dist_loss(conv1d_hat, y_hat.detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, y_hat, None, y_hat_lgt, None, info
