import os
import pickle
import cv2
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


class CSLDailyDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None):
        """
        初始化数据集。

        :param data_dir: 数据集根目录路径。
        :param split_file: 划分文件路径，包含训练/验证/测试集的视频ID列表。
        :param transform: 可选的图像转换函数。
        """
        self.data_dir = data_dir
        self.split_file = split_file
        self.transform = transform

        # 加载数据集划分
        with open(os.path.join(data_dir, 'sentence_label', split_file), 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]

        # 加载词汇到ID的映射
        with open(os.path.join(data_dir, 'gloss2ids.pkl'), 'rb') as f:
            self.gloss_to_id = pickle.load(f)

        # 加载视频ID到路径的映射
        with open(os.path.join(data_dir, 'sentence_label', 'video_map.txt'), 'r') as f:
            self.video_path_map = {}
            for line in f:
                vid, path = line.strip().split(',')
                self.video_path_map[vid] = path

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。

        :param idx: 样本索引。
        :return: 处理后的图像和对应的标签。
        """
        # 获取当前样本的视频ID
        video_id = self.video_ids[idx]

        # 构建视频帧路径
        video_path = os.path.join(self.data_dir, 'sentence_frames-512x512', 'frames_512x512',
                                  self.video_path_map[video_id])

        # 加载视频帧
        frame_files = sorted(os.listdir(video_path))
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(video_path, frame_file))
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)
        frames = np.stack(frames, axis=0)  # (num_frames, height, width, channels)

        # 加载标签
        label_file = os.path.join(self.data_dir, 'csl-daily-keypoints.pkl')
        with open(label_file, 'rb') as f:
            keypoints_data = pickle.load(f)
        # 假设标签数据已经按照视频ID排序
        label = keypoints_data[video_id]
        # 将标签转换为ID
        label_ids = [self.gloss_to_id[gloss] for gloss in label]

        return frames, label_ids

    @staticmethod
    def collate_fn(batch):
        """
        自定义的 collate 函数用于处理一批数据。

        :param batch: 一批数据，每个元素是一个 (frames, labels) 元组。
        :return: 处理后的批次数据。
        """
        frames, labels = zip(*batch)

        # 假设 frames 已经是 (num_frames, height, width, channels) 形式的张量
        # 使用 pad_sequence 来处理不同长度的视频帧
        frames = pad_sequence([torch.from_numpy(f).permute(3, 0, 1, 2) for f in frames], batch_first=True)

        # 处理标签
        labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        # 假设所有样本的标签长度相同，如果不同则需要使用 pad_sequence
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # 使用 -1 作为填充值

        return frames, labels


# 示例使用
data_dir = 'path/to/csl-daily'
split_file = 'csl-daily.train'  # 或 'csl-daily.dev', 'csl-daily.test'
dataset = CSLDailyDataset(data_dir, split_file)

# 创建 DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

# 测试 DataLoader
for batch in dataloader:
    frames, labels = batch
    print("Frames shape:", frames.shape)
    print("Labels shape:", labels.shape)
    break
