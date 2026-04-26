from abc import abstractmethod
from typing import Any, Union, Sequence

import torch
from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing_extensions import override

from ..BaseModel import SLRTBaseModel


class BaseModel(SLRTBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'KpsBaseModel'

    @abstractmethod
    def _init_network(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _define_loss_function(self):
        pass

    @abstractmethod
    def step_forward(self, batch) -> Any:
        pass

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

    @override
    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """
        Configures and returns a list of callback functions.

        This method does not take any parameters.

        :return: A list containing instances of EarlyStopping and ModelCheckpoint callbacks.
        """
        # Configure early stopping callback to monitor validation loss and stop training when it stops decreasing
        early_stop = EarlyStopping(
            monitor='Val/Loss',
            mode='min',
            **self.hparams.callback["EarlyStopping"]
        )

        # Configure model checkpoint callback to monitor validation word error rate and save the best model
        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_save_dir,
            monitor='Val/Word-Error-Rate',
            mode='min',
            **self.hparams.callback["ModelCheckpoint"]
        )

        # Return the configured list of callback functions
        return [early_stop, checkpoint]
