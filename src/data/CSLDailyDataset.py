import os
import pickle

import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.transforms import Compose, ToTensor


# TODO: Implement the CSLDailyDataset class
class CSLDailyDataset(Dataset):
    def __init__(self, features_path, annotation_file, split_file, mode, gloss2ids_file, transform=None):
        """

        """
        self.features_path = os.path.abspath(features_path)
        self.annotation_file = os.path.abspath(annotation_file)
        self.gloss2ids_file = os.path.abspath(gloss2ids_file)
        self.split_file = os.path.abspath(split_file)
        self.mode = mode

        self.annotation = self.get_annotations()
        self.gloss_dict = self.get_gloss_dict()

        self.transform = transform

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.annotation)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。

        :param idx: 样本索引。
        :return: 处理后的图像和对应的标签。
        """
        item = self.annotation[idx]

        # 构建视频帧路径
        frame_list_path = os.path.join(self.features_path, item['name'])
        if not frame_list_path:
            raise ValueError(f"No frames found for folder {frame_list_path}")

        # 加载视频帧
        frame_files = sorted(os.listdir(frame_list_path))
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(frame_list_path, frame_file))
            frames.append(frame)
        frames = np.stack(frames, axis=0)  # (num_frames, height, width, channels)
        if self.transform is None:
            self.transform = Compose([ToTensor()])
        frames, _ = self.transform(frames, [])
        frames = frames.float() / 127.5 - 1

        # 加载 标签
        label_ids = torch.LongTensor([self.gloss_dict[gloss] for gloss in item['label_gloss']])

        return frames, label_ids, item

    def get_annotations(self, sep='|'):
        """
        根据给定的模式和分割文件，从注解文件中筛选出符合条件的数据。

        :param annotation_file: 文件路径，包含注解信息
        :param split_file: 文件路径，包含用于筛选的模式信息
        :param mode: 模式列表，用于筛选样本
        :param sep: 分割符，默认为'|'
        :return: 筛选后的数据列表
        """
        # 尝试打开注解文件并加载数据
        try:
            with open(self.annotation_file, 'rb') as f:
                annotations = pickle.load(f)
        except Exception as e:
            # 若发生异常，打印错误信息并返回空列表
            print(f"Error loading annotation file: {e}")
            return []

        # 尝试打开分割文件并读取内容，分割每行数据
        try:
            with open(self.split_file, 'r') as file:
                lines = [line.strip().split(sep) for line in file]
        except Exception as e:
            # 若发生异常，打印错误信息并返回空列表
            print(f"Error loading split file: {e}")
            return []

        # 根据模式筛选样本名称
        sample_names = [line[0] for line in lines if line[1] in self.mode]
        # 根据样本名称筛选注解数据
        filtered_data = [item for item in annotations['info'] if item['name'] in sample_names]

        assert len(sample_names) == len(filtered_data)
        # 返回筛选后的数据列表
        return filtered_data

    def get_gloss_dict(self):
        """
        从指定的文件中加载pickle序列化的字典数据。

        :param gloss2ids_file: 字典文件路径
        :type gloss2ids_file: str
        :return: 加载的数据字典
        :rtype: dict
        """
        try:
            with open(self.gloss2ids_file, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print("文件未找到: ", self.gloss2ids_file)
            return None
        except Exception as e:
            print("出现错误:", str(e))
            return None
        else:
            return data

    @staticmethod
    def collate_fn(batch):
        """
        自定义的 collate 函数用于处理一批数据。

        :param batch: 一批数据，每个元素是一个 (frames, labels) 元组。
        :return: 处理后的批次数据。
        """
        frames, labels, info = zip(*batch)

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
