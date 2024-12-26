import os

from typing_extensions import override

from .BasePatchKpsDataModule import BasePatchKpsDataModule
from ...Datasets.PatchKpsDatasets import Phoenix2014TPatchKpsDataset


class Phoenix2014TPatchKpsDataModule(BasePatchKpsDataModule):
    """
    Phoenix2014PatchKpsDataModule is a PyTorch Lightning DataModule for the Phoenix2014PatchKpsDataset.

    Args:
        dataset_dir (str): The directory of the dataset.
        features_dir (str): The directory of the features.
        annotations_dir (str): The directory of the annotations.
        keypoints_file (str): The file of the keypoints.
        mode (stror list): The mode of the dataset.
    """

    def __init__(
            self,
            dataset_dir: str = None,
            features_dir: str = None,
            annotations_dir: str = None,
            keypoints_file: str = None,
            batch_size: int = 2,
            num_workers: int = 8,
            transform: [callable, dict] = None,
            tokenizer: [dict] = None,
            patch_hw: tuple[int, int] = (13, 13)
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, transform=transform, tokenizer=tokenizer)

        # Ensure all directory paths are set correctly
        self.dataset_dir = os.path.abspath(dataset_dir) if dataset_dir else None
        self.features_dir = os.path.abspath(features_dir) if features_dir else None
        self.annotations_dir = os.path.abspath(annotations_dir) if annotations_dir else None
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
        return Phoenix2014TPatchKpsDataset(
            dataset_dir=self.dataset_dir,
            features_dir=self.features_dir,
            annotations_dir=self.annotations_dir,
            keypoints_file=self.keypoints_file,
            mode=mode,
            transform={"video": video_transform, "keypoint": kps_transform},
            recognition_tokenizer=self.recognition_tokenizer,
            translation_tokenizer=self.translation_tokenizer,
            patch_hw=self.patch_hw
        )
