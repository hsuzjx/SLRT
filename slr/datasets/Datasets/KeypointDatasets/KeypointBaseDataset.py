import os
import pickle
from abc import abstractmethod

import torch
from torch.utils.data import Dataset

from slr.datasets.Datasets.utils import pad_label_sequence, pad_keypoints_sequence


class KeypointBaseDataset(Dataset):
    """
    """

    def __init__(
            self,
            keypoints_file: str = None,
            transform: callable = None,
            tokenizer: object = None,
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
        self.tokenizer = tokenizer

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
        glosses = self.__get_glosses(item)

        if self.transform:
            kps = self.transform(kps)
        else:
            kps = torch.from_numpy(kps)
        if self.tokenizer:
            glosses = self.tokenizer.encode(glosses)

        return kps, glosses, name

    @abstractmethod
    def _get_glosses(self, item) -> [str, list]:
        pass

    def collate_fn(self, batch):
        """
        Collates a list of samples into a batch.

        Args:
            batch (list): List of samples returned by `__getitem__`.

        Returns:
            tuple: Batched data including videos, labels, video lengths, label lengths, and info.
        """
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        kps, label, info = list(zip(*batch))

        kps, kps_length = pad_keypoints_sequence(kps, batch_first=True, num_keypoints=133)
        kps_length = torch.LongTensor(kps_length)
        label, label_length = pad_label_sequence(label, batch_first=True,
                                                 padding_value=self.tokenizer.convert_tokens_to_ids(
                                                     self.tokenizer.pad_token))
        label_length = torch.LongTensor(label_length)

        return kps, label, kps_length, label_length, info
