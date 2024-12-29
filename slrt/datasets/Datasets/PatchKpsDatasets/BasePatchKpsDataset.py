import glob
import os
from abc import abstractmethod
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import LiteralString

from ..utils import pad_video_sequence, pad_label_sequence, pad_keypoints_sequence


class BasePatchKpsDataset(Dataset):
    """

    """

    def __init__(
            self,
            transform: dict[str, callable] = None,
            recognition_tokenizer: object = None,
            translation_tokenizer: object = None,
            patch_hw: tuple[int, int] = (13, 13)
    ):
        super().__init__()

        self.features_dir = None
        self.annotations_dir = None
        self.keypoints_file = None
        self.mode = None

        self.video_info: pd.DataFrame = None
        self.kps_info = None
        self.frame_size = None

        # Set transform and tokenizer
        self.video_transform = transform["video"]
        self.kps_transform = transform["keypoint"]

        self.recognition_tokenizer = recognition_tokenizer
        self.translation_tokenizer = translation_tokenizer

        self.half_patch_h = (patch_hw[0] - 1) // 2
        self.half_patch_w = (patch_hw[1] - 1) // 2

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        if self.video_info is None:
            raise ValueError("Dataset info is not initialized.")

        item = self.video_info.iloc[idx]

        frames = self.__read_frames(item)  # (T, H, W, C)
        if self.video_transform:
            frames = self.video_transform(frames)

        kps = self.kps_info[item.name]["keypoints"]

        assert len(frames) == kps.shape[0]

        patchs_list = []
        for t in range(kps.shape[0]):
            frame = frames[t]
            x_max = frame.shape[1]
            y_max = frame.shape[0]

            frame = np.pad(
                frame,
                ((self.half_patch_h, self.half_patch_h), (self.half_patch_w, self.half_patch_w), (0, 0)),
                mode='constant',
                constant_values=0
            )

            frame_patches = []
            for v in range(kps.shape[1]):
                kp = kps[t, v]
                x, y, c = kp
                x = int(x)
                y = int(y)
                if 0 <= x < x_max and 0 <= y < y_max:
                    frame_patches.append(
                        frame[y:y + 2 * self.half_patch_h + 1, x:x + 2 * self.half_patch_w + 1, :]
                    )
                else:
                    frame_patches.append(
                        np.zeros((2 * self.half_patch_h + 1, 2 * self.half_patch_w + 1, 3), dtype=np.uint8)
                    )
            frame_patches = np.stack(frame_patches, axis=0)
            patchs_list.append(frame_patches)
        video_patches = np.stack(patchs_list, axis=0)  # (T, V, H, W, C)
        video_patches = torch.from_numpy(video_patches).permute(0, 4, 1, 2, 3)  # (T, C, V, H, W)
        video_patches = video_patches.float()

        if isinstance(kps, np.ndarray):
            kps = torch.from_numpy(kps)
        kps = kps.permute(2, 0, 1).contiguous()  # (T,V,C) -> (C,T,V)
        if self.kps_transform:
            kps = self.kps_transform(kps)

        glosses = self._get_glosses(item)
        translation = self._get_translation(item)

        glosses = self.recognition_tokenizer.encode(glosses) \
            if self.recognition_tokenizer and glosses else None
        translation = self.translation_tokenizer.encode(translation) \
            if self.translation_tokenizer and translation else None

        return video_patches, kps, glosses, translation, item

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
        patches, kps, glosses, translation, info = list(zip(*batch))

        patches, patches_length = pad_video_sequence(patches, batch_first=True, padding_value=0.0)
        patches_length = torch.LongTensor(patches_length)

        kps, kps_length = pad_keypoints_sequence(kps, batch_first=True, num_keypoints=133)
        kps_length = torch.LongTensor(kps_length)

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

        return patches, kps, gloss_label, translation_label, patches_length, kps_length, gloss_label_length, translation_label_length, info
