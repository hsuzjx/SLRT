import glob
import os
from abc import abstractmethod
from typing import Union

import cv2
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from typing_extensions import LiteralString

from slrt.datasets.Datasets.utils import pad_video_sequence, pad_label_sequence
from slrt.datasets.transforms import ToTensor


class BaseDataset(Dataset):
    """
    """

    def __init__(
            self,
            transform: callable = Compose([ToTensor()]),
            recognition_tokenizer: object = None,
            translation_tokenizer: object = None,
            read_hdf5: bool = False
    ):
        """
        """
        super().__init__()

        self.features_dir = None
        self.annotations_dir = None
        self.mode = None
        self.info: pd.DataFrame = None

        # Set transform and tokenizer
        self.transform = transform

        self.recognition_tokenizer = recognition_tokenizer
        self.translation_tokenizer = translation_tokenizer

        self.read_hdf5 = read_hdf5

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        if self.info is None:
            raise ValueError("Dataset info is not initialized.")
        return len(self.info)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the processed image and corresponding label.
        """
        if self.info is None:
            raise ValueError("Dataset info is not initialized.")

        item = self.info.iloc[idx]

        if self.read_hdf5:
            # frames = torch.load(os.path.join(frames_dir, 'video.pt'))
            with h5py.File(os.path.join(
                    self.features_dir,
                    self._get_frames_subdir_filename(item, filename=False),
                    "video.h5"
            ), 'r') as f:
                frames = f['data'][:]
            frames = torch.from_numpy(frames)
        else:
            frames = self.__read_frames(item)

        if self.transform:
            frames = self.transform(frames)

        glosses = self._get_glosses(item)
        translation = self._get_translation(item)

        glosses = self.recognition_tokenizer.encode(glosses) \
            if self.recognition_tokenizer and glosses else None
        translation = self.translation_tokenizer.encode(translation) \
            if self.translation_tokenizer and translation else None

        return frames, glosses, translation, item

    @abstractmethod
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        pass

    @abstractmethod
    def _get_glosses(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return None

    @abstractmethod
    def _get_translation(
            self,
            item: pd.DataFrame
    ) -> [str, list]:
        return None

    def __read_frames(self, item):
        """
        Reads video frames from the specified directory.

        Args:
            item (pd.DataFrame):

        Returns:
            list: List of frames read from the directory.
        """
        frames_file_list = sorted(
            glob.glob(os.path.join(
                self.features_dir,
                self._get_frames_subdir_filename(item, filename=True)
            )))
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
        video, glosses, translation, info = list(zip(*batch))

        video, video_length = pad_video_sequence(video, batch_first=True, padding_value=0.0)
        video_length = torch.LongTensor(video_length)

        if not None in glosses:
            gloss_label, gloss_label_length = pad_label_sequence(
                glosses, batch_first=True,
                padding_value=self.recognition_tokenizer.convert_tokens_to_ids(self.recognition_tokenizer.pad_token)
            )
            gloss_label_length = torch.LongTensor(gloss_label_length)
        else:
            gloss_label, gloss_label_length = None, None

        if not None in translation:
            translation_label, translation_label_length = pad_label_sequence(
                translation, batch_first=True,
                padding_value=self.translation_tokenizer.convert_tokens_to_ids(self.translation_tokenizer.pad_token)
            )
            translation_label_length = torch.LongTensor(translation_label_length)
        else:
            translation_label, translation_label_length = None, None

        info = [item.name for item in info]

        return video, gloss_label, translation_label, video_length, gloss_label_length, translation_label_length, info
