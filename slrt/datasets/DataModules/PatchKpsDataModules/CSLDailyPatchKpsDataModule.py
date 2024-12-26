import os

from typing_extensions import override

from .BasePatchKpsDataModule import BasePatchKpsDataModule
from ...Datasets.PatchKpsDatasets import CSLDailyPatchKpsDataset


class CSLDailyPatchKpsDataModule(BasePatchKpsDataModule):
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
            keypoints_file: str = None,
            batch_size: int = 2,
            num_workers: int = 8,
            transform: [callable, dict] = None,
            tokenizer: [dict] = None,
            patch_hw: tuple[int, int] = (13, 13)
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
        self.keypoints_file = os.path.abspath(keypoints_file) if keypoints_file else None

        self.patch_hw = patch_hw

    @override
    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            CSLDailyDataset: The dataset instance for the specified mode.
        """
        video_transform = self.video_transforms.get(mode, self.video_transforms['test'])
        kps_transform = self.kps_transforms.get(mode, self.kps_transforms['test'])
        return CSLDailyPatchKpsDataset(
            dataset_dir=self.dataset_dir,
            features_dir=self.features_dir,
            annotations_dir=self.annotations_dir,
            split_file=self.split_file,
            keypoints_file=self.keypoints_file,
            mode=mode,
            transform={"video": video_transform, "keypoint": kps_transform},
            recognition_tokenizer=self.recognition_tokenizer,
            translation_tokenizer=self.translation_tokenizer,
            patch_hw=self.patch_hw
        )
