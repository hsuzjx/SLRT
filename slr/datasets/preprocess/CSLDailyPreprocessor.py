import os
import pickle
from typing import Union

import pandas as pd
from typing_extensions import LiteralString, override

from slr.datasets.preprocess.BasePreprocessor import BasePreprocessor


class CSLDailyPreprocessor(BasePreprocessor):
    def __init__(self, dataset_dir, features_dir=None, annotations_dir=None, split_file=None):
        super().__init__(dataset_dir, features_dir, annotations_dir)
        self.name = "CSL-Daily"

        # Ensure all directory paths are set correctly
        self.features_dir = os.path.join(dataset_dir, 'sentence_frames-512x512/frames_512x512') \
            if features_dir is None and dataset_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(dataset_dir, 'sentence_label') \
            if annotations_dir is None and dataset_dir is not None else os.path.abspath(
            annotations_dir) if annotations_dir else None
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        # Set and validate split file
        self.split_file = os.path.join(self.annotations_dir, "split_1.txt") \
            if split_file is None else os.path.abspath(split_file)
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}")

        # Load and filter samples based on mode
        splits = pd.read_csv(split_file, sep='|', header=0, index_col='name')
        if "S000005_P0004_T00" in splits.index and "S000007_P0003_T00" not in splits.index:
            splits = splits.rename(index={"S000005_P0004_T00": "S000007_P0003_T00"})

        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info']).set_index('name')

        self.info = pd.concat([info, splits], axis=1)

    @override
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        subdir = item.name
        if filename:
            return os.path.join(subdir, "*.jpg")
        return subdir

    @override
    def _check_recognization(self) -> bool:
        return True

    @override
    def _check_translation(self) -> bool:
        return True

    @override
    def _get_glosses(self, item: pd.Series) -> list[str]:
        return item["label_gloss"]

    @override
    def _get_translation(self, item: pd.Series) -> list[str]:
        return item["label_word"]

    @override
    def _get_singer(self, item: pd.Series) -> str:
        return item['signer']

    def _get_postags(self, item: pd.Series) -> list[str]:
        return item['label_postag']

    def _get_chars(self, item: pd.Series) -> list[str]:
        return item['label_char']
