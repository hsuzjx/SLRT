import os
from typing import override

from slr.datasets.CSLDailyDataset import CSLDailyDataset
from slr.datasets.KeypointDatasets.CSLDailyKeypointDataset import CSLDailyKeypointDataset
from slr.datasets.KeypointDatasets.KeypointBaseDataModule import KeypointBaseDataModule


class CSLDailyDataModule(KeypointBaseDataModule):
    """
    """

    def __init__(
            self,
            keypoints_file: str = None,
            split_file: str = None,
            batch_size: int = 2,
            num_workers: int = 8,
            transform: [callable, dict] = None,
            tokenizer: [object, dict] = None,
    ):
        """
        """
        super().__init__(batch_size=batch_size, num_workers=num_workers, transform=transform, tokenizer=tokenizer)

        # Ensure all directory paths are set correctly
        self.keypoints_file = os.path.abspath(keypoints_file) if keypoints_file else None
        self.split_file = os.path.abspath(split_file) if split_file else None

    @override
    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            CSLDailyDataset: The dataset instance for the specified mode.
        """
        transform = self.transforms.get(mode, None)
        tokenizer = self.tokenizers.get(mode, None)
        return CSLDailyKeypointDataset(
            keypoints_file=self.keypoints_file,
            split_file=self.split_file,
            mode=mode,
            transform=transform,
            tokenizer=tokenizer
        )
