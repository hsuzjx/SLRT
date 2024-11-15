from abc import abstractmethod

import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, CenterCrop

from slr.datasets.BaseDataset import BaseDataset
from slr.datasets.transforms import ToTensor


class BaseDataModule(L.LightningDataModule):
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

        self.batch_size = self.hparams.get("batch_size", 2)
        self.num_workers = self.hparams.get("num_workers", 8)

        # Process transformations
        self.transforms = self.__process_transforms(self.hparams.get("transform", None))

        # Process tokenizers
        self.tokenizers = self.__process_tokenizers(self.hparams.get("tokenizer", None))

    @abstractmethod
    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            BaseDataset: The dataset instance for the specified mode.
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

    @staticmethod
    def __process_transforms(transform):
        """
        Processes the transform parameter into a dictionary.

        Args:
            transform ([callable, dict]): Transformations to apply to the dataset.

        Returns:
            dict: A dictionary of transformations for each dataset mode.
        """
        if transform is None:
            print("Warning: 'transform' is None, using default values.")
            return {
                'train': Compose(
                    [ToTensor(), Resize(256), RandomCrop(224), RandomHorizontalFlip()]  # , TemporalRescale(224)]
                ),
                'dev': Compose([ToTensor(), Resize(256), CenterCrop(224)]),
                'test': Compose([ToTensor(), Resize(256), CenterCrop(224)])
            }
        elif isinstance(transform, dict):
            keys = ['train', 'dev', 'test']
            for key in keys:
                if key not in transform:
                    print(f"Warning: '{key}' key missing from transform dict, setting to None.")
            return {key: transform.get(key, None) for key in keys}
        elif isinstance(transform, Compose):
            return {'train': transform, 'dev': transform, 'test': transform}
        else:
            raise ValueError("Invalid transform type. Expected dict or Compose instance.")

    @staticmethod
    def __process_tokenizers(tokenizer):
        """
        Processes the tokenizer parameter into a dictionary.

        Args:
            tokenizer ([object, dict]): Tokenizer to use for text processing.

        Returns:
            dict: A dictionary of tokenizers for each dataset mode.
        """
        if tokenizer is None:
            print("Warning: 'tokenizer' is None, using default values.")
            return {'train': None, 'dev': None, 'test': None}
        elif isinstance(tokenizer, dict):
            keys = ['train', 'dev', 'test']
            for key in keys:
                if key not in tokenizer:
                    print(f"Warning: '{key}' key missing from tokenizer dict, setting to None.")
            return {key: tokenizer.get(key, None) for key in keys}
        elif tokenizer is not None:
            return {'train': tokenizer, 'dev': tokenizer, 'test': tokenizer}
        else:
            raise ValueError("Invalid tokenizer type. Expected dict or Tokenizer object.")
