import os

from typing_extensions import override

from slr.datasets.DataModules.KeypointDataModules.KeypointBaseDataModule import KeypointBaseDataModule
from slr.datasets.Datasets.KeypointDatasets.Phoenix2014KeypointDataset import Phoenix2014KeypointDataset


class Phoenix2014KeypointDataModule(KeypointBaseDataModule):
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
            Phoenix2014KeypointDataset: The dataset instance for the specified mode.
        """
        transform = self.transforms.get(mode, None)
        tokenizer = self.tokenizers
        return Phoenix2014KeypointDataset(
            keypoints_file=self.keypoints_file,
            mode=mode,
            transform=transform,
            tokenizer=tokenizer
        )
