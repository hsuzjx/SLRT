from typing import Any, Tuple

import torch
from typing_extensions import override

from slrt.models.BaseModel import SLRTBaseModel
from slrt.models.XModel.modules import ResNet34


class XModel(SLRTBaseModel):
    """
    XModel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.name = "XModel"

    @override
    def _init_network(self, **kwargs):
        visual_backbone = ResNet34(
            pretrained=True,
            model_dir='../.pretrained_models',
            num_classes=self.recognition_tokenizer.vocab_size
        )

        visual_head = None

        temporal_module = None

    @override
    def _define_loss_function(self) -> Any:
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def step_forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]) -> Tuple[
        torch.Tensor, Any, Any, Any, Any, Any]:
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
                self.hparams.loss_weights[0] * self.ctc_loss(conv1d_hat.log_softmax(-1), y, y_hat_lgt, y_lgt).mean() +
                self.hparams.loss_weights[1] * self.ctc_loss(y_hat.log_softmax(-1), y, y_hat_lgt, y_lgt).mean() +
                self.hparams.loss_weights[2] * self.dist_loss(conv1d_hat, y_hat.detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, y_hat, None, y_hat_lgt, None, info

    @override
    def configure_optimizers(self):
        return None
