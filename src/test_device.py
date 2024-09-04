# author: muzhan
# contact: levio.pku@gmail.com
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
        data = torch.ones(size, size).float()
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
        print('batch train time:', time.time() - s_time, 's')

    # empty cache
    # torch.npu.empty_cache()


if __name__ == '__main__':
    train(15000, 1000)
