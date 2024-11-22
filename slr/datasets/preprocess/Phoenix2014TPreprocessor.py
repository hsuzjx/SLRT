import argparse
import os
from typing import Union

import pandas as pd
from typing_extensions import LiteralString, override

from BasePreprocessor import BasePreprocessor


class Phoenix2014TPreprocessor(BasePreprocessor):
    def __init__(self, dataset_dir, features_dir=None, annotations_dir=None):
        super().__init__(dataset_dir, features_dir, annotations_dir)
        self.name = "Phoenix2014T"

        # Ensure all directory paths are set correctly
        self.features_dir = os.path.join(dataset_dir, 'PHOENIX-2014-T/features/fullFrame-210x260px') \
            if features_dir is None and dataset_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(dataset_dir, 'PHOENIX-2014-T/annotations/manual') \
            if annotations_dir is None and dataset_dir is not None else os.path.abspath(
            annotations_dir) if annotations_dir else None
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        # Load corpus information
        corpus = {
            "train": pd.read_csv(os.path.join(self.annotations_dir, 'PHOENIX-2014-T.train.corpus.csv'),
                                 sep='|', header=0, index_col='name'),
            "dev": pd.read_csv(os.path.join(self.annotations_dir, 'PHOENIX-2014-T.dev.corpus.csv'),
                               sep='|', header=0, index_col='name'),
            "test": pd.read_csv(os.path.join(self.annotations_dir, 'PHOENIX-2014-T.test.corpus.csv'),
                                sep='|', header=0, index_col='name')
        }
        self.info = pd.concat([corpus[m].assign(split=m) for m in ["train", "dev", "test"]], axis=0, ignore_index=False)

    @override
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        subdir = os.path.join(item["split"], item.name)
        if filename:
            return os.path.join(subdir, "*.png")
        return subdir

    @override
    def _check_recognization(self) -> bool:
        return True

    @override
    def _check_translation(self) -> bool:
        return True

    @override
    def _get_glosses(self, item: pd.Series) -> list[str]:
        return [gloss for gloss in item['orth'].split(' ') if gloss]

    @override
    def _get_translation(self, item: pd.Series) -> list[str]:
        return [word for word in item['translation'].split(' ') if word]

    @override
    def _get_singer(self, item: pd.Series) -> str:
        return item['speaker']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=str,
                        default="../../../data/phoenix2014T",
                        help="Path to the dataset directory")
    parser.add_argument("--output-dir", type=str,
                        default="../../../data/preprocessed/phoenix2014T",
                        help="Output directory")

    args = parser.parse_args()

    processor = Phoenix2014TPreprocessor(dataset_dir=os.path.abspath(args.dataset_dir))

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    processor.resize_frames(output_dir=output_dir, dsize=(256, 256), max_workers=8)
    processor.generate_gloss_vocab(output_dir=output_dir)
    processor.generate_glosses_groundtruth(output_dir=output_dir)
    processor.generate_word_vocab(output_dir=output_dir)
    processor.generate_translation_groundtruth(output_dir=output_dir)
