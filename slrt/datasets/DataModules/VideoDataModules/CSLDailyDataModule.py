import os

from typing_extensions import override

from slrt.datasets.DataModules.VideoDataModules.BaseDataModule import BaseDataModule
from slrt.datasets.Datasets.VideoDatasets.CSLDailyDataset import CSLDailyDataset


class CSLDailyDataModule(BaseDataModule):
    """
    Data module for handling the CSL-Daily dataset within a PyTorch Lightning environment.

    This module encapsulates the dataset loading and preprocessing logic, including setting up
    different splits for training, validation, and testing, as well as applying transformations
    and tokenization as required.

    Attributes:
        dataset_dir (str): Path to the root directory of the dataset.
        features_dir (str): Path to the directory containing feature files.
        annotations_dir (str): Path to the directory containing annotation files.
        split_file (str): Path to the file that defines dataset splits.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of subprocesses to use for data loading.
        transforms (dict): Dictionary of transformations for each dataset mode.
        tokenizers (dict): Dictionary of tokenizers for each dataset mode.
        train_dataset (CSLDailyDataset): Training dataset instance.
        dev_dataset (CSLDailyDataset): Validation dataset instance.
        test_dataset (CSLDailyDataset): Test dataset instance.
    """

    def __init__(
            self,
            dataset_dir: str = None,
            features_dir: str = None,
            annotations_dir: str = None,
            split_file: str = None,
            batch_size: int = 2,
            num_workers: int = 8,
            transform: [callable, dict] = None,
            tokenizer: [object, dict] = None,
            read_hdf5: bool = False
    ):
        """
        Initializes the CSLDailyDataModule with the specified parameters.

        Args:
            dataset_dir (str): Path to the root directory of the dataset.
            features_dir (str): Path to the directory containing feature files.
            annotations_dir (str): Path to the directory containing annotation files.
            split_file (str): Path to the file that defines dataset splits.
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
        self.split_file = os.path.abspath(split_file) if split_file else None

        self.read_hdf5 = read_hdf5

    @override
    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            CSLDailyDataset: The dataset instance for the specified mode.
        """
        transform = self.transforms.get(mode, self.transforms['test'])
        tokenizer = self.tokenizers.get(mode, self.tokenizers['test'])
        return CSLDailyDataset(
            dataset_dir=self.dataset_dir,
            features_dir=self.features_dir,
            annotations_dir=self.annotations_dir,
            split_file=self.split_file,
            mode=mode,
            transform=transform,
            tokenizer=tokenizer,
            read_hdf5=self.read_hdf5
        )
