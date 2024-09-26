import os.path

import lightning as L
from torch.utils.data import DataLoader

from slr.datasets.CSLDailyDataset import CSLDailyDataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop
from slr.datasets.transforms import ToTensor, TemporalRescale


class CSLDailyDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir=None, features_dir=None, annotation_dir=None, split_file=None, batch_size=2,
                 num_workers=8,
                 train_transform=Compose(
                     [RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2)]),
                 dev_transform=Compose([CenterCrop(224), ToTensor()]),
                 test_transform=Compose([CenterCrop(224), ToTensor()]),
                 tokenizer=None):
        '''

        :param dataset_dir:
        :param features_dir:
        :param annotation_dir:
        :param split_file:
        :param batch_size:
        :param num_workers:
        :param train_transform:
        :param dev_transform:
        :param test_transform:
        '''
        self.dataset_dir = os.path.abspath(dataset_dir) if dataset_dir is not None else None
        self.features_dir = os.path.abspath(features_dir) if features_dir is not None else None
        self.annotation_dir = os.path.abspath(annotation_dir) if annotation_dir is not None else None
        self.split_file = os.path.abspath(split_file) if split_file is not None else None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.transforms = {
            'train': train_transform,
            'dev': dev_transform,
            'test': test_transform
        }

        self.tokenizer = tokenizer

    def load_dataset(self, mode):
        return CSLDailyDataset(
            dataset_dir=self.dataset_dir,
            features_dir=self.features_dir,
            annotation_dir=self.annotation_dir,
            split_file=self.split_file,
            mode=mode,
            transform=self.transforms[mode],
            tokenizer=self.tokenizer
        )

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = self.load_dataset('train')
            self.dev_dataset = self.load_dataset('dev')
        if stage == 'validate':
            self.dev_dataset = self.load_dataset('dev')
        if stage == 'test':
            self.test_dataset = self.load_dataset('test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=True, drop_last=True, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, drop_last=True, collate_fn=self.dev_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, drop_last=True, collate_fn=self.test_dataset.collate_fn)
