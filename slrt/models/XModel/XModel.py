from typing import Any, Tuple, Union, Sequence

import torch
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from typing_extensions import override

from slrt.models.BaseModel import SLRTBaseModel
from slrt.models.CorrNet.modules import NormLinear
from slrt.models.XModel.modules import *
from slrt.models.XModel.modules.TemproalModules import BiLSTMLayer
from slrt.models.XModel.modules.UNet1D.unet_model import UNet1D


class XModel(SLRTBaseModel):
    """
    XModel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "XModel"

    @override
    def _init_network(self, **kwargs):
        self.visual_backbone = ResNet18(
            num_classes=1000,
            **self.hparams.network['ResNet18']
        )
        self.visual_backbone.fc = Identity()
        # self.visual_backbone = S3D(1000)

        # self.visual_head = None

        self.conv1d = TemporalConv(
            **self.hparams.network['conv1d'],
            num_classes=self.recognition_tokenizer.vocab_size
        )
        self.conv1d.fc = NormLinear(1024, self.recognition_tokenizer.vocab_size)

        self.t_unet = UNet1D(n_channels=512, n_classes=512)
        # self.t_unet.outc = Identity()
        self.t_unet_fc = None

        self.temporal_module = BiLSTMLayer(
            **self.hparams.network['BiLSTM']
        )

        # self.classifier = self.conv1d.fc
        self.classifier = NormLinear(1024, self.recognition_tokenizer.vocab_size)

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
        x = videos.permute(0, 2, 1, 3, 4)
        x = self.visual_backbone(x)
        x = x.view(N, T, -1).permute(0, 2, 1)

        visual_features = self.t_unet(x)
        # lstm_output = self.temporal_module(visual_features.permute(2, 0, 1), video_lengths.to('cpu'))

        # predictions = lstm_output['predictions']
        # conv1d_logits = self.classifier(visual_features.transpose(1, 2)).transpose(1, 2)
        # output_logits = self.classifier(predictions)

        conv1d_output = self.conv1d(visual_features, video_lengths)
        visual_features = conv1d_output['visual_feat']
        feature_lengths = conv1d_output['feat_len']
        conv1d_logits = conv1d_output['conv_logits']
        lstm_output = self.temporal_module(visual_features, feature_lengths)

        # Get predictions
        predictions = lstm_output['predictions']
        # Pass through the classifier
        output_logits = self.classifier(predictions)

        return conv1d_logits, output_logits, feature_lengths

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
                25 * self.kl_loss(conv1d_hat, y_hat.detach(), use_blank=False)
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
        self.kl_loss = SeqKD(T=8)

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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
