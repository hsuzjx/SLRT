import os
import pickle
from abc import abstractmethod

import torch
from torch.utils.data import Dataset

from .utils import pad_label_sequence, pad_keypoints_sequence


class KeypointBaseDataset(Dataset):
    """
    """

    def __init__(
            self,
            keypoints_file: str = None,
            transform: callable = None,
            recognition_tokenizer: object = None,
            translation_tokenizer: object = None,
            frame_size: tuple = (210, 260)
    ):
        """
        """
        super().__init__()

        # Set keypoints file
        self.keypoints_file = os.path.abspath(keypoints_file)
        if not os.path.exists(self.keypoints_file):
            raise FileNotFoundError(f"Kenpoints file not found at {self.keypoints_file}")

        # Load keypoints info
        with open(self.keypoints_file, 'rb') as f:
            self.kps_info = pickle.load(f)

        self.kps_info_keys = sorted(self.kps_info.keys())

        # Set transform and tokenizer
        self.transform = transform

        self.recognition_tokenizer = recognition_tokenizer
        self.translation_tokenizer = translation_tokenizer

        self.frame_size = frame_size

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.kps_info_keys)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the processed image and corresponding label.
        """
        name = self.kps_info_keys[idx]
        item = self.kps_info[name]

        kps = item['keypoints']
        glosses = self._get_glosses(item)
        translation = self._get_translation(item)

        kps[:, :, 0] /= self.frame_size[0]
        kps[:, :, 1] = self.frame_size[1] - kps[:, :, 1]
        kps[:, :, 1] /= self.frame_size[1]
        kps[:, :, :2] = (kps[:, :, :2] - 0.5) / 0.5

        kps = torch.from_numpy(kps).permute(2, 0, 1)  # T,V,C -> C,T,V
        if self.transform:
            kps = self.transform(kps)

        glosses = self.recognition_tokenizer.encode(glosses) \
            if self.recognition_tokenizer and glosses else None
        translation = self.translation_tokenizer.encode(translation) \
            if self.translation_tokenizer and translation else None

        # return kps, glosses, translation, name
        return kps, glosses, translation, name

    @abstractmethod
    def _get_glosses(self, item) -> [str, list]:
        pass

    @abstractmethod
    def _get_translation(self, item) -> [str, list]:
        pass

    # def collate_fn(self, batch):
    #     """
    #     Collates a list of samples into a batch.
    #
    #     Args:
    #         batch (list): List of samples returned by `__getitem__`.
    #
    #     Returns:
    #         tuple: Batched data including videos, labels, video lengths, label lengths, and info.
    #     """
    #     batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
    #     kps, label_gloss, label_translation, name = list(zip(*batch))
    #
    #     kps, kps_length = pad_keypoints_sequence(kps, batch_first=True, num_keypoints=133)
    #     kps_length = torch.LongTensor(kps_length)
    #
    #     label_gloss, label_gloss_length = pad_label_sequence(
    #         label_gloss, batch_first=True,
    #         padding_value=self.recognition_tokenizer.convert_tokens_to_ids(self.recognition_tokenizer.pad_token)
    #     )
    #     label_gloss_length = torch.LongTensor(label_gloss_length)
    #
    #     label_translation, label_translation_length = pad_label_sequence(
    #         label_translation, batch_first=True,
    #         padding_value=self.translation_tokenizer.convert_tokens_to_ids(self.translation_tokenizer.pad_token)
    #     )
    #     label_translation_length = torch.LongTensor(label_translation_length)
    #
    #     return kps, label_gloss, label_translation, kps_length, label_gloss_length, label_translation_length, name

    def collate_fn(self, batch):
        """
        Collates a list of samples into a batch.

        Args:
            batch (list): List of samples returned by `__getitem__`.

        Returns:
            tuple: Batched data including videos, labels, video lengths, label lengths, and info.
        """
        batch = [item for item in sorted(batch, key=lambda x: x[0].shape[1], reverse=True)]
        kps, label_gloss, label_translation, name = list(zip(*batch))

        kps, kps_length = pad_keypoints_sequence(kps, batch_first=True, num_keypoints=133)
        kps_length = torch.LongTensor(kps_length)

        if not None in label_gloss:
            label_gloss, label_gloss_length = pad_label_sequence(
                label_gloss, batch_first=True,
                padding_value=self.recognition_tokenizer.convert_tokens_to_ids(self.recognition_tokenizer.pad_token)
            )
            label_gloss_length = torch.LongTensor(label_gloss_length)
        else:
            label_gloss, label_gloss_length = None, None

        if not None in label_translation:
            label_translation, label_translation_length = pad_label_sequence(
                label_translation, batch_first=True,
                padding_value=self.translation_tokenizer.convert_tokens_to_ids(self.translation_tokenizer.pad_token)
            )
            label_translation_length = torch.LongTensor(label_translation_length)
        else:
            label_translation, label_translation_length = None, None

        # return {
        #     "kps": kps,
        #     "glosses": label_gloss,
        #     "translation": label_translation,
        #     "kps_length": kps_length,
        #     "glosses_length": label_gloss_length,
        #     "translation_length": label_translation_length,
        #     "name": name
        # }

        return kps, label_gloss, label_translation, kps_length, label_gloss_length, label_translation_length, name
