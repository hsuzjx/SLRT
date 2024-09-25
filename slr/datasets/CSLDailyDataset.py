import glob
import os
import pickle

import cv2
import pandas as pd
from torch.utils.data import Dataset


class CSLDailyDataset(Dataset):
    """
    Custom Dataset class for loading and processing the CSL Daily dataset.

    This class is responsible for loading feature and annotation data based on provided paths.
    It also handles data splitting according to the specified mode (e.g., train, dev, test).
    Additionally, it supports data augmentation and tokenization.

    Args:
        dataset_dir (str): Base directory of the dataset.
        features_dir (str): Directory containing feature files. If not provided, defaults to a subdirectory within `dataset_dir`.
        annotation_dir (str): Directory containing annotation files. If not provided, defaults to a subdirectory within `dataset_dir`.
        split_file (str): Path to the data split file. If not provided, defaults to "split_1.txt" in the annotation directory.
        mode (str): Specifies the dataset mode, which can be "train", "dev", or "test".
        transform (callable): Optional data transformation function for data augmentation or normalization.
        tokenizer (object): Tokenizer object for text tokenization.
    """

    def __init__(
            self,
            dataset_dir: str,
            features_dir: str = None,
            annotation_dir: str = None,
            split_file: str = None,
            mode: str = "train",
            transform: callable = None,
            tokenizer: object = None
    ):
        # Set default or custom path for feature directory and check its validity
        if features_dir is None and dataset_dir is not None:
            self.features_dir = os.path.join(
                dataset_dir, 'sentence_frames-512x512/frames_512x512'
            )
        else:
            self.features_dir = os.path.abspath(features_dir) if features_dir is not None else None

        # Set default or custom path for annotation directory and check its validity
        if annotation_dir is None and dataset_dir is not None:
            self.annotations_dir = os.path.join(
                dataset_dir, 'sentence_label'
            )
        else:
            self.annotations_dir = os.path.abspath(annotation_dir) if annotation_dir is not None else None

        # Validate paths
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        # Validate the mode and set it
        if mode not in ["train", "dev", "test"]:
            raise ValueError("mode must be one of 'train', 'dev', or 'test'")
        self.mode = mode

        # Set path for split file and validate its existence
        if split_file is not None:
            self.split_file = os.path.abspath(split_file)
        else:
            self.split_file = os.path.join(self.annotations_dir, "split_1.txt")
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}")

        # Load data from split file and filter samples based on mode
        splits = pd.read_csv(self.split_file, sep='|', header=0)
        self.sample_list = splits[splits['split'] == mode]['name'].tolist()

        # Handle special cases for default split_1.txt in training mode
        if split_file is None and mode == "train":
            if "S000005_P0004_T00" in self.sample_list:
                self.sample_list.remove("S000005_P0004_T00")
            if "S000007_P0003_T00" not in self.sample_list:
                self.sample_list.append("S000007_P0003_T00")

        # Load annotation file and filter out information for current mode
        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")

        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info'])
        self.info = info[info['name'].isin(self.sample_list)]

        # Set data transformation and tokenization
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns the length of self.info to indicate the number of samples in the dataset.
        """
        return len(self.info)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the specified index.

        :param idx: Sample index.
        :return: Processed image and corresponding label.
        """
        item = self.info.iloc[idx]

        # Build the path to video frames
        frames_dir = os.path.join(self.features_dir, item['name'])
        # Check if the frames directory exists
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found at {frames_dir}")
        frames = self._read_frames(frames_dir)

        glosses = item['label_gloss']

        if self.transform is not None:
            frames = self.transform(frames)
        if self.tokenizer is not None:
            glosses = self.tokenizer.encode(glosses)

        return frames, glosses, item

    def _read_frames(self, frames_dir):
        """
        Reads video frames from the specified directory and returns them as a list.

        :param frames_dir: Directory containing video frames.
        :return: List of frames read from the directory.
        """
        frames_file_list = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
        frames = []

        for frame_file in frames_file_list:
            frame = cv2.imread(frame_file)
            if frame is None:
                raise ValueError(f"Failed to read frame from {frame_file}")
            # TODO: Confirm the correct frame's channels
            frames.append(frame)

        return frames
