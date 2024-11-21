import unittest

import numpy as np

from slr.datasets.DataModules.VideoDataModules.Phoenix2014DataModule import Phoenix2014DataModule

# 定义数据特征和注释的路径以及手势词典的路径
FEATURES_PATH = '../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px'
ANNOTATIONS_PATH = '../data/phoenix2014/phoenix-2014-multisigner/annotations/manual'
GLOSS_DICT = np.load('../data/global_files/gloss_dict/phoenix2014_gloss_dict.npy',
                     allow_pickle=True).item()

# 测试类，用于测试Phoenix2014DataModule的各个功能
class TestPhoenix2014DataModule(unittest.TestCase):

    def setUp(self):
        # 初始化数据模块
        self.data_module = Phoenix2014DataModule(
            features_path=FEATURES_PATH,
            annotations_path=ANNOTATIONS_PATH,
            gloss_dict=GLOSS_DICT,
            batch_size=2,
            num_workers=8
        )

    def test_setup_fit(self):
        # 测试setup方法在'fit'阶段的表现
        self.data_module.setup(stage='fit')
        self.assertIsNotNone(self.data_module.train_dataset)
        self.assertIsNotNone(self.data_module.dev_dataset)

    def test_train_dataloader(self):
        # 测试train_dataloader方法，确保其能够返回训练数据加载器
        self.data_module.setup(stage='fit')
        train_loader = self.data_module.train_dataloader()
        self.assertIsNotNone(train_loader)

    def test_val_dataloader(self):
        # 测试val_dataloader方法，确保其能够返回验证数据加载器
        self.data_module.setup(stage='fit')
        val_loader = self.data_module.val_dataloader()
        self.assertIsNotNone(val_loader)

    def test_setup_test(self):
        # 测试setup方法在'test'阶段的表现
        self.data_module.setup(stage='test')
        self.assertIsNotNone(self.data_module.test_dataset)

    def test_test_dataloader(self):
        # 测试test_dataloader方法，确保其能够返回测试数据加载器
        self.data_module.setup(stage='test')
        test_loader = self.data_module.test_dataloader()
        self.assertIsNotNone(test_loader)


if __name__ == '__main__':
    unittest.main()
