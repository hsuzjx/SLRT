import argparse
import os
import pickle
from typing import Union

import pandas as pd
from typing_extensions import LiteralString, override

from BasePatchPreprocessor import BasePatchPreprocessor


class Phoenix2014PatchPreprocessor(BasePatchPreprocessor):
    def __init__(self, dataset_dir, kps_file, features_dir=None, annotations_dir=None):
        super().__init__()
        self.name = "Phoenix2014"

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

        # Load corpus information
        corpus = {
            "train": pd.read_csv(os.path.join(self.annotations_dir, 'train.corpus.csv'),
                                 sep='|', header=0, index_col='id'),
            "dev": pd.read_csv(os.path.join(self.annotations_dir, 'dev.corpus.csv'),
                               sep='|', header=0, index_col='id'),
            "test": pd.read_csv(os.path.join(self.annotations_dir, 'test.corpus.csv'),
                                sep='|', header=0, index_col='id')
        }
        self.info = pd.concat([corpus[m].assign(split=m) for m in ["train", "dev", "test"]], axis=0, ignore_index=False)

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
        subdir = os.path.join(item["split"], item.name, "1")
        if filename:
            return os.path.join(subdir, "*.png")
        return subdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=str,
                        default="../../../../data/phoenix2014",
                        help="Path to the dataset directory")
    parser.add_argument("--kps-file", type=str,
                        default="../../../../data/preprocessed/phoenix2014/phoenix2014-keypoints.pkl",
                        help="Path to the keypoints file")
    parser.add_argument("--output-dir", type=str,
                        default="../../../../data/preprocessed/phoenix2014",
                        help="Output directory")

    args = parser.parse_args()

    processor = Phoenix2014PatchPreprocessor(
        dataset_dir=os.path.abspath(args.dataset_dir),
        kps_file=os.path.abspath(args.kps_file)
    )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    processor.ge_patches(output_dir=output_dir, patch_hw=(13, 13), max_workers=8)
