import os
import pickle
from typing import Union

import pandas as pd
from typing_extensions import override, LiteralString

from .BasePatchKpsDataset import BasePatchKpsDataset


class Phoenix2014PatchKpsDataset(BasePatchKpsDataset):
    def __init__(
            self,
            dataset_dir: str = None,
            features_dir: str = None,
            annotations_dir: str = None,
            keypoints_file: str = None,
            mode: [str, list] = "train",
            transform: callable = None,
            recognition_tokenizer: object = None,
            translation_tokenizer: object = None,
            patch_hw: tuple[int, int] = (13, 13)
    ):
        super().__init__(
            transform=transform,
            recognition_tokenizer=recognition_tokenizer,
            translation_tokenizer=translation_tokenizer,
            patch_hw=patch_hw
        )

        # Ensure all directory paths are set correctly
        self.features_dir = os.path.join(dataset_dir, 'phoenix-2014-multisigner/features/fullFrame-210x260px') \
            if features_dir is None and dataset_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(dataset_dir, 'phoenix-2014-multisigner/annotations/manual') \
            if annotations_dir is None and dataset_dir is not None else os.path.abspath(
            annotations_dir) if annotations_dir else None
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        # Convert mode to list and validate
        self.mode = [mode] if isinstance(mode, str) else mode
        if not all(m in ["train", "dev", "test"] for m in self.mode):
            raise ValueError("Each element in mode must be one of 'train', 'dev', or 'test'")

        # Load corpus information
        corpus = {
            "train": pd.read_csv(os.path.join(self.annotations_dir, 'train.corpus.csv'),
                                 sep='|', header=0, index_col='id') if "train" in self.mode else None,
            "dev": pd.read_csv(os.path.join(self.annotations_dir, 'dev.corpus.csv'),
                               sep='|', header=0, index_col='id') if "dev" in self.mode else None,
            "test": pd.read_csv(os.path.join(self.annotations_dir, 'test.corpus.csv'),
                                sep='|', header=0, index_col='id') if "test" in self.mode else None
        }
        self.video_info = pd.concat([corpus[m].assign(split=m) for m in self.mode], axis=0, ignore_index=False)

        # Special handling for training mode
        if "train" in self.mode:
            if "13April_2011_Wednesday_tagesschau_default-14" in self.video_info.index:
                self.video_info.drop("13April_2011_Wednesday_tagesschau_default-14", axis=0, inplace=True)

        # Set keypoints file
        self.keypoints_file = os.path.abspath(keypoints_file)
        if not os.path.exists(self.keypoints_file):
            raise FileNotFoundError(f"Kenpoints file not found at {self.keypoints_file}")

        # Load keypoints info
        with open(self.keypoints_file, 'rb') as f:
            self.kps_info = pickle.load(f)

    @override
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        if filename:
            return os.path.join(item["split"], item.name, "1", "*.png")
        return os.path.join(item["split"], item.name, "1")

    @override
    def _get_glosses(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return [gloss for gloss in item['annotation'].split(' ') if gloss]

    @override
    def _get_translation(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return None
