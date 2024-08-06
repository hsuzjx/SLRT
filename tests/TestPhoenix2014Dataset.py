import unittest

import numpy as np
import torch
from src.data.Phoenix2014Dataset import Phoenix2014Dataset


class TestPhoenix2014Dataset(unittest.TestCase):

    def setUp(self):
        # 根据你的项目路径替换下面的路径
        features_path = '../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px'
        annotations_path = '../data/phoenix2014/phoenix-2014-multisigner/annotations/manual'
        gloss_dict = np.load('../data/global_files/gloss_dict/phoenix2014_gloss_dict.npy',
                             allow_pickle=True).item()
        self.dataset = Phoenix2014Dataset(features_path, annotations_path, gloss_dict, mode='train')

    def test_length(self):
        # 测试数据集的长度是否正确
        self.assertTrue(len(self.dataset) > 0)

    def test_get_item(self):
        # 测试__getitem__方法
        for i in range(len(self.dataset)):
            imgs, label_list, name = self.dataset[i]
            self.assertTrue(isinstance(imgs, torch.Tensor))
            self.assertTrue(isinstance(label_list, torch.Tensor))
            self.assertTrue(isinstance(name, str))

    # 添加更多的测试用例根据需求


if __name__ == '__main__':
    unittest.main()
