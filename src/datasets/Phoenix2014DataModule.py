import os.path

import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .Phoenix2014Dataset import Phoenix2014Dataset
from .transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, \
    TemporalRescale, ToTensor


class Phoenix2014DataModule(L.LightningDataModule):

    def __init__(self, features_path, annotations_path, gloss_dict, batch_size=2, num_workers=8,
                 train_transform=Compose(
                     [RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2), ]),
                 dev_transform=Compose([CenterCrop(224), ToTensor(), ]),
                 test_transform=Compose([CenterCrop(224), ToTensor(), ])
                 ):
        super().__init__()
        self.features_path = os.path.abspath(features_path)
        self.annotations_path = os.path.abspath(annotations_path)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.gloss_dict = gloss_dict

        # datasets
        self.test_dataset = None
        self.dev_dataset = None
        self.train_dataset = None

        # transforms
        self.train_transform = train_transform
        self.dev_transform = dev_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = Phoenix2014Dataset(features_path=self.features_path,
                                                    annotations_path=self.annotations_path,
                                                    gloss_dict=self.gloss_dict,
                                                    mode="train",
                                                    transform=self.train_transform)
            self.dev_dataset = Phoenix2014Dataset(features_path=self.features_path,
                                                  annotations_path=self.annotations_path,
                                                  gloss_dict=self.gloss_dict,
                                                  mode="dev",
                                                  transform=self.train_transform)

        if stage == 'test':
            self.test_dataset = Phoenix2014Dataset(features_path=self.features_path,
                                                   annotations_path=self.annotations_path,
                                                   gloss_dict=self.gloss_dict,
                                                   mode="test",
                                                   transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.train_dataset.collate_fn, pin_memory=True, drop_last=True, )

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.dev_dataset.collate_fn, pin_memory=True, drop_last=True, )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.test_dataset.collate_fn, pin_memory=True, drop_last=True, )
