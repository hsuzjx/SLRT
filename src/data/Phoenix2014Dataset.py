import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from .transforms import Compose, ToTensor


class Phoenix2014Dataset(data.Dataset):
    """
    Phoenix2014数据集类，继承自PyTorch的Dataset类。

    参数:
    - features_path: 特征文件路径
    - annotations_path: 注释文件路径
    - gloss_dict: 手势词汇字典
    - mode: 数据集模式，"train"或"test"或"dev"
    - transform: 数据变换，如果为None，则使用默认变换
    """

    def __init__(self, features_path, annotations_path, gloss_dict, mode="train", drop_ids=None, transform=None):
        super().__init__()
        self.mode = mode
        self.features_path = os.path.abspath(features_path)
        self.annotations_path = os.path.abspath(annotations_path)
        self.dict = gloss_dict

        corpus_file_path = os.path.join(self.annotations_path, f'{self.mode}.corpus.csv')
        try:
            self.corpus = pd.read_csv(corpus_file_path, sep='|', header=0, index_col='id')
        except FileNotFoundError:
            raise FileNotFoundError(f"Corpus file not found at {corpus_file_path}")

        # Drop specific ID if needed
        if drop_ids is not None:
            for drop_id in drop_ids:
                if drop_id in self.corpus.index:
                    self.corpus.drop(drop_id, axis=0, inplace=True)

        if transform is None:
            self.transform = Compose([ToTensor()])
        else:
            self.transform = transform

    def __getitem__(self, idx):
        """
        获取数据集中的第idx项数据。

        参数:
        - idx: 数据索引

        返回:
        - imgs: 图像数据
        - label_list: 标签列表
        - item.name: 数据名称
        """
        item = self.corpus.iloc[idx]
        img_list_path = sorted(glob.glob(os.path.join(self.features_path, f'{self.mode}', item.folder)))
        if not img_list_path:
            raise ValueError(f"No images found for folder {item.folder}")

        anno = item.annotation.split(' ')
        anno = [word for word in anno if word]  # 移除空字符串
        label_list = [self.dict.get(w, 0) for w in anno]  # 默认为0，如果单词不在字典中

        imgs = []
        for img_path in img_list_path:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found or failed to load: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        if self.transform:
            imgs, label_list = self.transform(imgs, label_list)

        imgs = imgs.float() / 127.5 - 1
        label_list = torch.LongTensor(label_list)

        return imgs, label_list, item

    def __len__(self):
        """
        返回数据集的大小。

        返回:
        - 数据集大小
        """
        return len(self.corpus)

    @staticmethod
    def collate_fn(batch):
        """
        动态padding函数，用于将一批数据进行padding对齐。
    
        参数:
        - batch: 一批数据
    
        返回:
        - padded_video: 填充后的视频数据
        - video_length: 每个视频的原始长度
        - padded_label: 填充后的标签数据（如果存在）
        - label_length: 每个标签的原始长度
        - info: 数据的额外信息
        """
        # 按视频长度降序排序批次数据
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        # 解压缩batch，分别获取视频数据、标签数据和额外信息
        video, label, info = list(zip(*batch))
        # 对视频数据进行padding处理
        if len(video[0].shape) > 3:
            # 找到批次中最长的视频长度
            max_len = len(video[0])
            # 计算每个视频的padding后长度
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            # 左侧padding大小
            left_pad = 6
            # 右侧padding大小
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            # 更新最大长度以包括padding
            max_len = max_len + left_pad + right_pad
            # 对每个视频进行padding
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            # 将padding后的视频数据堆叠成张量
            padded_video = torch.stack(padded_video)
        else:
            # 找到批次中最长的视频长度
            max_len = len(video[0])
            # 计算每个视频的原始长度
            video_length = torch.LongTensor([len(vid) for vid in video])
            # 对每个视频进行padding
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            # 将padding后的视频数据堆叠成张量并转换维度
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        # 计算每个标签的长度
        label_length = torch.LongTensor([len(lab) for lab in label])
        # 根据是否有标签数据进行处理
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            # 对标签数据进行padding
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info
