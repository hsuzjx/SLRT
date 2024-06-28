import glob
import os
import time
import warnings

import pandas as pd
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch.utils.data as data
from .utils.data_augmentation import *


# sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, features_path, annotations_path, gloss_dict=None,
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

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

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


train_transform = Compose([RandomCrop(224),
                           RandomHorizontalFlip(0.5),
                           ToTensor(),
                           TemporalRescale(0.2), ])
dev_transform = Compose([CenterCrop(224),
                         ToTensor(), ])
test_transform = Compose([CenterCrop(224),
                          ToTensor(), ])

if __name__ == "__main__":
    print(os.path.abspath(os.path.curdir))
    gloss_dict_path = './.tmp'
    if not os.path.exists(gloss_dict_path):
        os.makedirs(gloss_dict_path)
    feeder = BaseFeeder(features_path='./phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px',
                        annotations_path='./phoenix2014-release/phoenix-2014-multisigner/annotations/manual',
                        gloss_dict_path='./.tmp',
                        ground_truth_path='./.tmp',
                        mode='train',
                        transform=train_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        collate_fn=feeder.collate_fn,
        pin_memory=True
    )

    for epoch in range(10):
        for data in tqdm(dataloader):
            pass
