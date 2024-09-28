import os

import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, CenterCrop

from slr.datasets.Phoenix2014TDataset2 import Phoenix2014TDataset
from slr.datasets.transforms import ToTensor


class Phoenix2014TDataModule(L.LightningDataModule):
    """
    Data module for handling the Phoenix 2014 T dataset within a PyTorch Lightning environment.

    This module encapsulates the dataset loading and preprocessing logic, including setting up
    different splits for training, validation, and testing, as well as applying transformations
    and tokenization as required.

    Attributes:
        dataset_dir (str): Path to the root directory of the dataset.
        features_dir (str): Path to the directory containing feature files.
        annotations_dir (str): Path to the directory containing annotation files.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of subprocesses to use for data loading.
        transforms (dict): Dictionary of transformations for each dataset mode.
        tokenizers (dict): Dictionary of tokenizers for each dataset mode.
        train_dataset (Phoenix2014TDataset): Training dataset instance.
        dev_dataset (Phoenix2014TDataset): Validation dataset instance.
        test_dataset (Phoenix2014TDataset): Test dataset instance.
    """

    def __init__(
            self,
            dataset_dir: str = None,
            features_dir: str = None,
            annotations_dir: str = None,
            batch_size: int = 2,
            num_workers: int = 8,
            transform: [callable, dict] = None,
            tokenizer: [object, dict] = None
    ):
        """
        Initializes the Phoenix2014TDataModule with the specified parameters.

        Args:
            dataset_dir (str): Path to the root directory of the dataset.
            features_dir (str): Path to the directory containing feature files.
            annotations_dir (str): Path to the directory containing annotation files.
            batch_size (int): Batch size for dataloaders. Defaults to 2.
            num_workers (int): Number of subprocesses to use for data loading. Defaults to 8.
            transform ([callable, dict], optional): Transformations to apply to the dataset.
                Can be a callable or a dictionary of callables for different modes. Defaults to None.
            tokenizer ([object, dict], optional): Tokenizer to use for text processing.
                Can be a tokenizer object or a dictionary of tokenizer objects for different modes. Defaults to None.
        """
        super().__init__()

        # Ensure all directory paths are set correctly
        self.dataset_dir = os.path.abspath(dataset_dir) if dataset_dir else None
        self.features_dir = os.path.abspath(features_dir) if features_dir else None
        self.annotations_dir = os.path.abspath(annotations_dir) if annotations_dir else None

        # Configuration parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize datasets
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        # Process transformations
        self.transforms = self._process_transforms(transform)

        # Process tokenizers
        self.tokenizers = self._process_tokenizers(tokenizer)

    def _process_transforms(self, transform):
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

    def _process_tokenizers(self, tokenizer):
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

    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            Phoenix2014TDataset: The dataset instance for the specified mode.
        """
        transform = self.transforms.get(mode, self.transforms['test'])
        tokenizer = self.tokenizers.get(mode, self.tokenizers['test'])
        return Phoenix2014TDataset(
            dataset_dir=self.dataset_dir,
            features_dir=self.features_dir,
            annotations_dir=self.annotations_dir,
            mode=mode,
            transform=transform,
            tokenizer=tokenizer
        )

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=True, drop_last=True, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        """
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, drop_last=True, collate_fn=self.dev_dataset.collate_fn)

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, drop_last=True, collate_fn=self.test_dataset.collate_fn)
