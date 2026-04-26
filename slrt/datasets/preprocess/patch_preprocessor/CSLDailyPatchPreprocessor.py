import argparse
import os
import pickle
from typing import Union

import pandas as pd
from typing_extensions import override, LiteralString

from BasePatchPreprocessor import BasePatchPreprocessor


class CSLDailyPatchPreprocessor(BasePatchPreprocessor):
    def __init__(self, dataset_dir, kps_file, features_dir=None, annotations_dir=None, split_file=None):
        super().__init__()
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
        splits = pd.read_csv(self.split_file, sep='|', header=0, index_col='name')
        if "S000005_P0004_T00" in splits.index and "S000007_P0003_T00" not in splits.index:
            splits = splits.rename(index={"S000005_P0004_T00": "S000007_P0003_T00"})

        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info']).set_index('name')

        self.info = pd.concat([info, splits], axis=1)

        self.gloss_map = data['gloss_map']
        self.char_map = data['char_map']
        self.word_map = data['word_map']
        self.postag_map = data['postag_map']

        self.kps_file = os.path.abspath(kps_file)
        if not os.path.exists(self.kps_file):
            raise FileNotFoundError(f"Kenpoints file not found at {self.kps_file}")

        # Load keypoints info
        with open(self.kps_file, 'rb') as f:
            self.kps_info = pickle.load(f)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=str,
                        default="../../../../data/csl-daily",
                        help="Path to the dataset directory")
    parser.add_argument("--kps-file", type=str,
                        default="../../../../data/preprocessed/csl-daily/csl-daily-keypoints.pkl",
                        help="Path to the keypoints file")
    parser.add_argument("--output-dir", type=str,
                        default="../../../../data/preprocessed/csl-daily",
                        help="Output directory")

    args = parser.parse_args()

    processor = CSLDailyPatchPreprocessor(
        dataset_dir=os.path.abspath(args.dataset_dir),
        kps_file=os.path.abspath(args.kps_file)
    )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    processor.ge_patches(output_dir=output_dir, patch_hw=(13, 13), max_workers=8)
