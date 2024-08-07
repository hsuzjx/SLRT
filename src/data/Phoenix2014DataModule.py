import os.path

import lightning as L
from torch.utils.data import DataLoader

# 导入自定义的数据集和数据预处理模块
from .Phoenix2014Dataset import Phoenix2014Dataset
from .transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, \
    TemporalRescale, ToTensor


# 定义用于处理Phoenix2014数据集的数据模块
class Phoenix2014DataModule(L.LightningDataModule):

    def __init__(self, features_path, annotations_path, gloss_dict, batch_size=2, num_workers=8,
                 train_transform=Compose(
                     [RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2)]),
                 dev_transform=Compose([CenterCrop(224), ToTensor()]),
                 test_transform=Compose([CenterCrop(224), ToTensor()])):
        """
        初始化数据模块，设置路径、参数和变换。

        参数:
        features_path -- 特征文件的路径
        annotations_path -- 注释文件的路径
        gloss_dict -- 手势词汇表
        batch_size -- 批处理大小（默认2）
        num_workers -- 加载数据的工人数量（默认8）
        train_transform -- 训练数据的预处理流程
        dev_transform -- 验证数据的预处理流程
        test_transform -- 测试数据的预处理流程
        """
        super().__init__()

        # 确保路径是合法和存在的
        if not os.path.exists(features_path) or not os.path.exists(annotations_path):
            raise FileNotFoundError("Features or annotations path does not exist.")
        self.features_path = os.path.abspath(features_path)
        self.annotations_path = os.path.abspath(annotations_path)

        # 确保参数是合法的
        if batch_size <= 0 or num_workers <= 0:
            raise ValueError("Batch size and number of workers must be greater than 0.")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.gloss_dict = gloss_dict

        # 数据集初始化为空
        self.test_dataset = None
        self.dev_dataset = None
        self.train_dataset = None

        # 变换初始化
        self.train_transform = train_transform
        self.dev_transform = dev_transform
        self.test_transform = test_transform

    def load_dataset(self, mode, transform, drop_ids=None):
        """
        封装数据集加载逻辑，提高代码复用性。

        参数:
        mode -- 数据集的模式（"train", "dev", "test"）
        transform -- 数据预处理流程

        返回:
        数据集实例
        """
        try:
            return Phoenix2014Dataset(features_path=self.features_path,
                                      annotations_path=self.annotations_path,
                                      gloss_dict=self.gloss_dict,
                                      mode=mode,
                                      drop_ids=drop_ids,
                                      transform=transform)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset with mode {mode}: {e}")

    def setup(self, stage=None):
        """
        根据运行阶段加载相应的数据集。

        参数:
        stage -- 运行阶段（"fit", "test"）
        """
        if stage == 'fit':
            self.train_dataset = self.load_dataset("train", self.train_transform,
                                                   drop_ids=['13April_2011_Wednesday_tagesschau_default-14'])
            self.dev_dataset = self.load_dataset("dev", self.dev_transform)

        if stage == 'test':
            self.test_dataset = self.load_dataset("test", self.test_transform)

    def train_dataloader(self):
        """
        返回训练数据加载器。

        返回:
        DataLoader实例
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.train_dataset.collate_fn, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        """
        返回验证数据加载器。

        返回:
        DataLoader实例
        """
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.dev_dataset.collate_fn, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        """
        返回测试数据加载器。

        返回:
        DataLoader实例
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.test_dataset.collate_fn, pin_memory=True, drop_last=True)
