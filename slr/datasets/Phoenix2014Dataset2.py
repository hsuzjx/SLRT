import glob
import os

import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from slr.datasets.transforms import ToTensor
from slr.datasets.utils import pad_video_sequence, pad_label_sequence


class Phoenix2014Dataset(Dataset):
    """
    Phoenix 2014 dataset class for sign language recognition tasks.

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
            mode: str = "train",
            transform: callable = Compose([ToTensor()]),
            tokenizer: object = None
    ) -> None:
        """
        Initializes the Phoenix2014Dataset with the given parameters.

        Args:
            dataset_dir (str, optional): Base directory of the dataset.
            features_dir (str, optional): Directory containing video frame features.
                                          Defaults to a subdirectory within `dataset_dir`.
            annotations_dir (str, optional): Directory containing annotations.
                                             Defaults to a subdirectory within `dataset_dir`.
            mode (str, optional): The dataset mode ("train", "dev", or "test").
                                  Defaults to "train".
            transform (callable, optional): Transformation function applied to video frames.
            tokenizer (object, optional): Tokenizer used for encoding labels.
        """
        super().__init__()

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

        # mode must be one of 'train', 'dev', or 'test'
        self.mode = mode
        if self.mode not in ["train", "dev", "test"]:
            raise ValueError("Mode must be one of 'train', 'dev', or 'test'")

        # Load corpus information
        self.info = pd.read_csv(os.path.join(self.annotations_dir, f'{self.mode}.corpus.csv'),
                                sep='|', header=0, index_col='id')

        # Special handling for training mode
        if self.mode == "train":
            if "13April_2011_Wednesday_tagesschau_default-14" in self.info.index:
                self.info.drop("13April_2011_Wednesday_tagesschau_default-14", axis=0, inplace=True)

        # Set transform and tokenizer
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.info)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset given its index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing video frames, glosses, and additional info.
        """
        item = self.info.iloc[idx]
        frames_dir = os.path.join(self.features_dir, self.mode, item.name, "1")
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found at {frames_dir}")
        frames = self._read_frames(frames_dir)

        glosses = [gloss for gloss in item['annotation'].split(' ') if gloss]

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
        frames_file_list = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        frames = []
        for frame_file in frames_file_list:
            frame = cv2.imread(frame_file)
            if frame is None:
                raise ValueError(f"Failed to read frame from {frame_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames

    def collate_fn(self, batch):
        """
        Collates a list of samples into a batch.

        Args:
            batch (list): List of samples returned by `__getitem__`.

        Returns:
            tuple: Batched data including videos, labels, video lengths, label lengths, and info.
        """
        video, label, info = list(zip(*batch))

        video_length = [len(v) for v in video]
        video = pad_video_sequence(video, batch_first=True, padding_value=0.0)
        label_length = [len(l) for l in label]
        label = pad_label_sequence(label, batch_first=True,
                                   padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
        info = [item.name for item in info]

        return video, label, video_length, label_length, info
