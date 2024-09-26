import glob
import os
import pickle

import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from slr.datasets.utils import pad_video_sequence, pad_label_sequence
from slr.datasets.transforms import ToTensor


class CSLDailyDataset(Dataset):
    """
    Custom Dataset class for loading and processing the CSL Daily dataset.

    This class loads feature and annotation data based on provided paths, handles
    data splitting according to the specified mode (e.g., train, dev, test), and
    supports data augmentation and tokenization.

    Args:
        dataset_dir (str): Base directory of the dataset.
        features_dir (str, optional): Directory containing feature files. Defaults
                                      to a subdirectory within `dataset_dir`.
        annotation_dir (str, optional): Directory containing annotation files.
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

    def __init__(
            self,
            dataset_dir: str,
            features_dir: str = None,
            annotation_dir: str = None,
            split_file: str = None,
            mode: [str, list] = "train",
            transform: callable = Compose([ToTensor()]),
            tokenizer: object = None
    ):
        """
        Initializes the dataset with the given parameters.

        Validates input directories and initializes instance variables.
        Converts `mode` to a list if it is a string.
        Loads the dataset split and filters samples based on the mode.
        """
        # Set and validate feature directory
        self.features_dir = os.path.join(dataset_dir, 'sentence_frames-512x512/frames_512x512') \
            if features_dir is None and dataset_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(dataset_dir, 'sentence_label') \
            if annotation_dir is None and dataset_dir is not None else os.path.abspath(
            annotation_dir) if annotation_dir else None
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
        self.sample_list = splits[splits['split'].isin(self.mode)]['name'].tolist()

        # Special handling for training mode
        if split_file is None and "train" in self.mode:
            if "S000005_P0004_T00" in self.sample_list:
                self.sample_list.remove("S000005_P0004_T00")
            if "S000007_P0003_T00" not in self.sample_list:
                self.sample_list.append("S000007_P0003_T00")

        # Load annotations and filter by mode
        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info'])
        self.info = info[info['name'].isin(self.sample_list)]

        # Set transformations and tokenizer
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.info)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the processed image and corresponding label.
        """
        item = self.info.iloc[idx]
        frames_dir = os.path.join(self.features_dir, item['name'])
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found at {frames_dir}")
        frames = self._read_frames(frames_dir)

        glosses = item['label_gloss']
        if self.transform:
            frames = self.transform(frames)
        if self.tokenizer:
            glosses = self.tokenizer.encode(glosses)

        return frames, glosses, item

    def _read_frames(self, frames_dir):
        """
        Reads video frames from the specified directory.

        Args:
            frames_dir (str): Directory containing video frames.

        Returns:
            list: List of frames read from the directory.
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

    @staticmethod
    def collate_fn(batch):
        video, label, info = list(zip(*batch))

        video_length = [len(v) for v in video]
        video = pad_video_sequence(video, batch_first=True, padding_value=0.0)
        label_length = [len(l) for l in label]
        label = pad_label_sequence(label, batch_first=True, padding_value=0.0)
        info = [item['name'] for item in info]

        return video, label, video_length, label_length, info
