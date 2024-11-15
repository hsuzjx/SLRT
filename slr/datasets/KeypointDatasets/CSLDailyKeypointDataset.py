import os
from typing import override

import pandas as pd

from slr.datasets.KeypointDatasets.KeypointBaseDataset import KeypointBaseDataset


class CSLDailyKeypointDataset(KeypointBaseDataset):
    """
    """

    def __init__(
            self,
            keypoints_file: str = None,
            split_file: str = None,
            mode: [str, list] = "train",
            transform: callable = None,
            tokenizer: object = None,
    ):
        """
        """
        super().__init__(keypoints_file=keypoints_file, transform=transform, tokenizer=tokenizer)

        # Convert mode to list and validate
        self.mode = [mode] if isinstance(mode, str) else mode
        if not all(m in ["train", "dev", "test"] for m in self.mode):
            raise ValueError("Each element in mode must be one of 'train', 'dev', or 'test'")

        # Set and validate split file
        self.split_file = os.path.abspath(split_file)
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}")

        # Load and filter samples based on mode
        splits = pd.read_csv(self.split_file, sep='|', header=0)
        sample_list = splits[splits['split'].isin(self.mode)]['name'].tolist()

        # Special handling for training mode
        if "train" in self.mode:
            if "S000005_P0004_T00" in sample_list:
                sample_list.remove("S000005_P0004_T00")
            if "S000007_P0003_T00" not in sample_list:
                sample_list.append("S000007_P0003_T00")

        self.kps_info_keys = sorted(sample_list)

    @override
    def __get_glosses(self, item) -> [str, list]:
        return item['label_gloss']
