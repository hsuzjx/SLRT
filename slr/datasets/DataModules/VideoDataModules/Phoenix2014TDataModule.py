import os

from typing_extensions import override

from slr.datasets.DataModules.VideoDataModules.BaseDataModule import BaseDataModule
from slr.datasets.Datasets.VideoDatasets.Phoenix2014TDataset import Phoenix2014TDataset


class Phoenix2014TDataModule(BaseDataModule):
    """
    Data module for handling the Phoenix2014T dataset within a PyTorch Lightning environment.

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
            tokenizer: [object, dict] = None,
            read_hdf5: bool = False
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
        super().__init__(batch_size=batch_size, num_workers=num_workers, transform=transform, tokenizer=tokenizer)

        # Ensure all directory paths are set correctly
        self.dataset_dir = os.path.abspath(dataset_dir) if dataset_dir else None
        self.features_dir = os.path.abspath(features_dir) if features_dir else None
        self.annotations_dir = os.path.abspath(annotations_dir) if annotations_dir else None

        self.read_hdf5 = read_hdf5

    @override
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
            tokenizer=tokenizer,
            read_hdf5=self.read_hdf5
        )
