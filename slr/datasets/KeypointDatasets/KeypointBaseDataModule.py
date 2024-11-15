from abc import abstractmethod

import lightning as L
from torch.utils.data import DataLoader

from slr.datasets.KeypointDatasets.KeypointBaseDataset import KeypointBaseDataset


class KeypointBaseDataModule(L.LightningDataModule):
    """
    """

    def __init__(self, **kwargs):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize datasets
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        # Set hyperparameters
        self.batch_size = self.hparams.get("batch_size", 2)
        self.num_workers = self.hparams.get("num_workers", 8)

        # Process transformations
        self.transforms = self.hparams.get("transform", None)

        # Process tokenizers
        self.tokenizers = self.hparams.get("tokenizer", None)

    @abstractmethod
    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            KeypointBaseDataset: The dataset instance for the specified mode.
        """
        pass

    def setup(self, stage=None):
        """
        Prepare the datasets for training, validation, and testing.

        Args:
            stage (str, optional): The stage to set up ('fit', 'validate', 'test'). If None, set up all stages.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = self.load_dataset('train')
            self.dev_dataset = self.load_dataset('dev')
        if stage == 'validate' or stage is None:
            self.dev_dataset = self.load_dataset('dev')
        if stage == 'test' or stage is None:
            self.test_dataset = self.load_dataset('test')

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
            prefetch_factor=self.hparams.get("prefetch_factor", 2),
            persistent_workers=self.hparams.get("persistent_workers", False)
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        """
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.dev_dataset.collate_fn,
            prefetch_factor=self.hparams.get("prefetch_factor", 2),
            persistent_workers=self.hparams.get("persistent_workers", False)
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.test_dataset.collate_fn,
            prefetch_factor=self.hparams.get("prefetch_factor", 2),
            persistent_workers=self.hparams.get("persistent_workers", False)
        )
