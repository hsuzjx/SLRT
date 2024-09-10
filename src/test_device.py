import os
import sys
import time
import signal
import argparse
from tabnanny import verbose

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch_npu

from data.Phoenix2014Dataset import Phoenix2014Dataset
from data.transforms import *


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
    time1 = time.time()
    batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
    video, label, info = list(zip(*batch))
    time2 = time.time()
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
    time3 = time.time()
    label_length = torch.LongTensor([len(lab) for lab in label])
    time4 = time.time()
    if max(label_length) == 0:
        return padded_video, video_length, [], [], info
    else:
        padded_label = []
        for lab in label:
            padded_label.extend(lab)
        padded_label = torch.LongTensor(padded_label)
        time5 = time.time()
        print(time5 - time4, time4 - time3, time3 - time2, time2 - time1)
        return padded_video, video_length, padded_label, label_length, info


def test_dataset():
    ds = Phoenix2014Dataset(
        features_path='/home/ma-user/work/workspace/SLR/data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='/home/ma-user/work/workspace/SLR/data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
        gloss_dict=np.load('/home/ma-user/work/workspace/SLR/data/global_files/gloss_dict/phoenix2014_gloss_dict.npy',
                           allow_pickle=True).item(),
        mode='train',
        drop_ids=['13April_2011_Wednesday_tagesschau_default-14'],
        transform=Compose([RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2)])
    )

    dl = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, collate_fn=collate_fn, pin_memory=True, drop_last=True, num_workers=100
    )

    device = torch.device('npu:0')
    e_time = time.time()
    for i, (x, x_lgt, y, y_lgt, info) in tqdm(enumerate(dl), total=len(dl)):
        s_time = time.time()
        x = x.to(device)
        y = y.to(device)
        print('to device time:', time.time() - s_time, 's')
        print('all time:', time.time() - e_time, 's')
        e_time = time.time()


# from torch_npu.contrib import transfer_to_npu


class MyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def train(size, n_batches):
    # set device
    device = torch.device('npu:0')
    # 创建模型
    model = MyModel(size, size).to(device)
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 创建损失函数
    criterion = nn.MSELoss().to(device)

    model.train()

    # 创建一个进度条
    for _ in tqdm(range(n_batches), desc='Processing:', ):
        # data
        s_time = time.time()
        data = torch.ones(50, size, size).float()
        ge_time = time.time()
        print(f'data generate time (torch.ones({size}, {size}).float()):', ge_time - s_time, 's')
        data = data.to(device)
        print('to device time (data.to(device)):', time.time() - ge_time, 's')
        # data = torch.ones(size, size).float().to(device)
        # print('data generate & to device time:', time.time() - s_time, 's')

        s_time = time.time()
        # zero grad
        optimizer.zero_grad()

        # forward
        outputs = model(data)
        # calculate loss
        loss = criterion(outputs, data)
        # backward
        loss.backward()
        # step
        optimizer.step()
        # print('batch train time:', time.time() - s_time, 's')

    # empty cache
    # torch.npu.empty_cache()


if __name__ == '__main__':
    # train(1500, 1000)
    test_dataset()
