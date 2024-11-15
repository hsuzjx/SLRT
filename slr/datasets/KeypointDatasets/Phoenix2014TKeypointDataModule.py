import os
from typing import override

from slr.datasets.KeypointDatasets.KeypointBaseDataModule import KeypointBaseDataModule
from slr.datasets.KeypointDatasets.Phoenix2014TKeypointDataset import Phoenix2014TKeypointDataset


class Phoenix2014TKeypointDataModule(KeypointBaseDataModule):
    """
    """

    def __init__(
            self,
            keypoints_file: str = None,
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

    @override
    def load_dataset(self, mode):
        """
        Load the dataset for the specified mode with appropriate transformations and tokenization.

        Args:
            mode (str): The dataset mode ('train', 'dev', or 'test').

        Returns:
            Phoenix2014TKeypointDataset: The dataset instance for the specified mode.
        """
        transform = self.transforms.get(mode, None)
        tokenizer = self.tokenizers.get(mode, None)
        return Phoenix2014TKeypointDataset(
            keypoints_file=self.keypoints_file,
            mode=mode,
            transform=transform,
            tokenizer=tokenizer
        )
