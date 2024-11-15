import os
import pickle
from typing import override, Union, LiteralString

import pandas as pd
from torchvision.transforms import Compose

from slr.datasets.BaseDataset import BaseDataset
from slr.datasets.transforms import ToTensor


class CSLDailyDataset(BaseDataset):
    """
    Custom Dataset class for loading and processing the CSL Daily dataset.

    This class loads feature and annotation data based on provided paths, handles
    data splitting according to the specified mode (e.g., train, dev, test), and
    supports data augmentation and tokenization.

    Attributes:
        features_dir (str): Path to the directory containing feature files.
        annotations_dir (str): Path to the directory containing annotation files.
        split_file (str): Path to the file that defines dataset splits.
        mode (list): List of dataset modes ("train", "dev", or "test").
        transform (callable): Data transformation function for augmentation or normalization.
        tokenizer (object): Tokenizer object for text tokenization.
    """

    def __init__(
            self,
            dataset_dir: str,
            features_dir: str = None,
            annotations_dir: str = None,
            split_file: str = None,
            mode: [str, list] = "train",
            transform: callable = Compose([ToTensor()]),
            tokenizer: object = None,
            read_hdf5: bool = False
    ):
        """
        Initializes the dataset with the given parameters.

        Validates input directories and initializes instance variables.
        Converts `mode` to a list if it is a string.
        Loads the dataset split and filters samples based on the mode.

        Args:
            dataset_dir (str): Base directory of the dataset.
            features_dir (str, optional): Directory containing feature files. Defaults
                                          to a subdirectory within `dataset_dir`.
            annotations_dir (str, optional): Directory containing annotation files.
                                            Defaults to a subdirectory within `dataset_dir`.
            split_file (str, optional): Path to the data split file. Defaults to
                                        "split_1.txt" in the annotation directory.
            mode (str or list, optional): Specifies the dataset mode ("train", "dev",
                                          or "test"). Can be a single string or a list
                                          of strings. Defaults to "train".
            transform (callable, optional): Data transformation function for
                                            augmentation or normalization.
            tokenizer (object, optional): Tokenizer object for text tokenization.
        """
        super().__init__(transform=transform, tokenizer=tokenizer, read_hdf5=read_hdf5)

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

        # Convert mode to list and validate
        self.mode = [mode] if isinstance(mode, str) else mode
        if not all(m in ["train", "dev", "test"] for m in self.mode):
            raise ValueError("Each element in mode must be one of 'train', 'dev', or 'test'")

        # Set and validate split file
        self.split_file = os.path.join(self.annotations_dir, "split_1.txt") \
            if split_file is None else os.path.abspath(split_file)
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}")

        # Load and filter samples based on mode
        splits = pd.read_csv(self.split_file, sep='|', header=0)
        sample_list = splits[splits['split'].isin(self.mode)]['name'].tolist()

        # Special handling for training mode
        if split_file is None and "train" in self.mode:
            if "S000005_P0004_T00" in sample_list:
                sample_list.remove("S000005_P0004_T00")
            if "S000007_P0003_T00" not in sample_list:
                sample_list.append("S000007_P0003_T00")

        # Load annotations and filter by mode
        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info'])
        self.info = info[info['name'].isin(sample_list)]
        self.info.set_index("name", inplace=True)

    @override
    def __get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        if filename:
            return os.path.join(item.name, "*.jpg")
        else:
            return item.name

    @override
    def __get_glosses(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return item['label_gloss']
