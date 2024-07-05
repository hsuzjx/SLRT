import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from .transforms import Compose, ToTensor


# warnings.simplefilter(action='ignore', category=FutureWarning)


class Phoenix2014Dataset(data.Dataset):
    def __init__(self, features_path, annotations_path, gloss_dict,
                 mode="train",
                 transform=Compose([ToTensor()])):
        # super().__init__()
        self.mode = mode
        self.features_path = os.path.abspath(features_path)
        self.annotations_path = os.path.abspath(annotations_path)

        self.dict = gloss_dict

        self.corpus = pd.read_csv(os.path.join(self.annotations_path, f'{self.mode}.corpus.csv'),
                                  sep='|', header=0, index_col='id')
        if self.mode == 'train':
            self.corpus.drop('13April_2011_Wednesday_tagesschau_default-14', axis=0, inplace=True)
        self.transform = transform

    def __getitem__(self, idx):
        fi = self.corpus.iloc[idx]
        img_list = sorted(glob.glob(os.path.join(self.features_path, f'{self.mode}', fi.folder)))
        anno = fi.annotation.split(' ')
        while '' in anno:
            anno.remove('')
        label_list = [self.dict[w] for w in anno]

        imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
        label = label_list

        if self.transform is not None:
            imgs, label = self.transform(imgs, label)
        imgs = imgs.float() / 127.5 - 1
        label = torch.LongTensor(label)

        return imgs, label, fi.name

    def __len__(self):
        return len(self.corpus)

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info
