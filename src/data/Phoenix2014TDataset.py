import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from .transforms import Compose, ToTensor

class Phoenix2014TDataset(data.Dataset):
    def __init__(self, features_path, annotations_path, gloss_dict, mode="train", transform=None):
        """
        初始化数据集。

        参数:
        - features_path: 特征数据的路径
        - annotations_path: 注解数据的路径
        - gloss_dict: 手语词汇字典
        - mode: 数据集的模式（"train", "test"等）
        - transform: 应用于数据的可选变换
        """
        self.features_path = features_path
        self.annotations_path = annotations_path
        self.gloss_dict = gloss_dict
        self.mode = mode
        self.transform = transform

        # 加载数据集列表，这里假设有一个函数实现加载逻辑
        self.data_list = self._load_data_list()

    def _load_data_list(self):
        """
        加载数据集列表的私有方法。
        """
        # 这里仅作为示例，应根据实际数据集的组织方式来实现加载逻辑
        return [os.path.join(self.features_path, f) for f in os.listdir(self.features_path)]

    def __getitem__(self, idx):
        """
        获取数据集中的某一项。
        """
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError("Index out of range")

        # 读取特征数据
        feature_path = self.data_list[idx]
        features = np.load(feature_path)  # 假设特征数据为Numpy格式

        # 读取注解数据，这里简化逻辑，实际应根据数据格式来处理
        annotation_path = os.path.join(self.annotations_path, os.path.basename(feature_path).replace('.npy', '.csv'))
        annotations = pd.read_csv(annotation_path).values

        # 应用变换
        if self.transform:
            features, annotations = self.transform(features, annotations)

        return features, annotations

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        """
        自定义的合并批次数据的函数。
        """
        # 这里仅做简单示例，实际可能需要根据模型输入要求做更复杂的处理
        features, annotations = zip(*batch)
        return torch.tensor(features), torch.tensor(annotations)
