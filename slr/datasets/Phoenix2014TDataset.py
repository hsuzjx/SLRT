import glob
import os

import cv2
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from slr.datasets.transforms import ToTensor
from slr.datasets.utils import pad_video_sequence, pad_label_sequence


class Phoenix2014TDataset(Dataset):
    """
    Phoenix 2014 T dataset class for sign language recognition tasks.

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
            tokenizer: object = None,
            read_hdf5: bool = False
    ) -> None:
        """
        Initializes the Phoenix2014TDataset with the given parameters.

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

        # mode must be one of 'train', 'dev', or 'test'
        self.mode = mode
        if self.mode not in ["train", "dev", "test"]:
            raise ValueError("Mode must be one of 'train', 'dev', or 'test'")

        # Load corpus information
        self.info = pd.read_csv(os.path.join(self.annotations_dir, f'PHOENIX-2014-T.{self.mode}.corpus.csv'),
                                sep='|', header=0, index_col='name')

        # Special handling for training mode
        # if self.mode == "train":
        #     if "13April_2011_Wednesday_tagesschau_default-14" in self.info.index:
        #         self.info.drop("13April_2011_Wednesday_tagesschau_default-14", axis=0, inplace=True)

        # Set transform and tokenizer
        self.transform = transform
        self.tokenizer = tokenizer

        self.read_hdf5 = read_hdf5

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
        frames_dir = os.path.join(self.features_dir, self.mode, item.name)
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found at {frames_dir}")

        if self.read_hdf5:
            # frames = torch.load(os.path.join(frames_dir, 'video.pt'))
            with h5py.File(os.path.join(frames_dir, "video.h5"), 'r') as f:
                frames = f['data'][:]
            frames = torch.from_numpy(frames)
        else:
            frames = self._read_frames(frames_dir)

        glosses = [gloss for gloss in item['orth'].split(' ') if gloss]

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
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))

        video, video_length = pad_video_sequence(video, batch_first=True, padding_value=0.0)
        video_length = torch.LongTensor(video_length)
        label, label_length = pad_label_sequence(label, batch_first=True,
                                                 padding_value=self.tokenizer.convert_tokens_to_ids(
                                                     self.tokenizer.pad_token))
        label_length = torch.LongTensor(label_length)
        info = [item.name for item in info]

        return video, label, video_length, label_length, info
