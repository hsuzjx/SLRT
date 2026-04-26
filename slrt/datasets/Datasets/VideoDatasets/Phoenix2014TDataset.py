import os
from typing import Union

import pandas as pd
from torchvision.transforms import Compose
from typing_extensions import LiteralString
from typing_extensions import override

from slrt.datasets.Datasets.VideoDatasets.BaseDataset import BaseDataset
from slrt.datasets.transforms import ToTensor


class Phoenix2014TDataset(BaseDataset):
    """
    Phoenix2014T dataset class for sign language recognition tasks.

    This class provides functionality to load video frames and their corresponding
    annotations, applying transformations and tokenization as needed.

    Attributes:
        features_dir (str): Path to the directory containing video frame features.
        annotations_dir (str): Path to the directory containing annotations.
        mode (str): The dataset mode ("train", "dev", or "test").
        info (pd.DataFrame): DataFrame containing dataset information.
        transform (callable): Transformation function applied to video frames.
        tokenizer (object): Tokenizer used for encoding labels.
    """

    def __init__(
            self,
            dataset_dir: str = None,
            features_dir: str = None,
            annotations_dir: str = None,
            mode: [str, list] = "train",
            transform: callable = Compose([ToTensor()]),
            recognition_tokenizer: object = None,
            translation_tokenizer: object = None,
            read_hdf5: bool = False
    ):
        """
        Initializes the Phoenix2014TDataset with the given parameters.

        Args:
            dataset_dir (str, optional): Base directory of the dataset.
            features_dir (str, optional): Directory containing video frame features.
                                          Defaults to a subdirectory within `dataset_dir`.
            annotations_dir (str, optional): Directory containing annotations.
                                             Defaults to a subdirectory within `dataset_dir`.
            mode (str or list, optional): Specifies the dataset mode ("train", "dev",
                                          or "test"). Can be a single string or a list
                                          of strings. Defaults to "train".
            transform (callable, optional): Transformation function applied to video frames.
            tokenizer (object, optional): Tokenizer used for encoding labels.
        """
        super().__init__(
            transform=transform,
            recognition_tokenizer=recognition_tokenizer,
            translation_tokenizer=translation_tokenizer,
            read_hdf5=read_hdf5
        )

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

        # Convert mode to list and validate
        self.mode = [mode] if isinstance(mode, str) else mode
        if not all(m in ["train", "dev", "test"] for m in self.mode):
            raise ValueError("Each element in mode must be one of 'train', 'dev', or 'test'")

        # Load corpus information
        corpus = {
            "train": pd.read_csv(os.path.join(self.annotations_dir, 'PHOENIX-2014-T.train.corpus.csv'),
                                 sep='|', header=0, index_col='name') if "train" in self.mode else None,
            "dev": pd.read_csv(os.path.join(self.annotations_dir, 'PHOENIX-2014-T.dev.corpus.csv'),
                               sep='|', header=0, index_col='name') if "dev" in self.mode else None,
            "test": pd.read_csv(os.path.join(self.annotations_dir, 'PHOENIX-2014-T.test.corpus.csv'),
                                sep='|', header=0, index_col='name') if "test" in self.mode else None
        }
        self.info = pd.concat([corpus[m].assign(split=m) for m in self.mode], axis=0, ignore_index=False)

        # Special handling for training mode
        # if self.mode == "train":
        #     if "13April_2011_Wednesday_tagesschau_default-14" in self.info.index:
        #         self.info.drop("13April_2011_Wednesday_tagesschau_default-14", axis=0, inplace=True)

    @override
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        if filename:
            return os.path.join(item["split"], item.name, "*.png")
        return os.path.join(item["split"], item.name)

    @override
    def _get_glosses(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return [gloss for gloss in item['orth'].split(' ') if gloss]

    @override
    def _get_translation(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return [word for word in item['translation'].split(' ') if word]
