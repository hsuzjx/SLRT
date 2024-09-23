import os

import numpy as np
from omegaconf import DictConfig

from slr.datasets import Phoenix2014DataModule
from slr.datasets.preprocess import preprocess


def init_datamodule(dataset_name, features_path, annotations_path, gloss_dict_path, ground_truth_path,
                    datamodule_cfg: DictConfig):
    """
    根据提供的数据集名称和配置，设置数据模块。

    :param dataset_name: 字符串，表示数据集的名称。
    :param features_path: 字符串，表示特征文件的路径。
    :param annotations_path: 字符串，表示注释文件的路径。
    :param gloss_dict_path: 字符串，表示词汇表文件的路径。
    :param ground_truth_path: 字符串，表示地面真实文件的路径。
    :param datamodule_cfg: DictConfig 对象，包含数据模块的配置。
    :return: 返回相应数据集的数据模块对象，如果数据集不支持则抛出异常。
    """

    # TODO: 处理 Phoenix2014T 和 CSL-Daily 数据集的逻辑
    if dataset_name == 'phoenix2014':
        try:
            # 尝试对数据进行预处理
            preprocess(
                dataset_name=dataset_name,
                annotations_path=annotations_path,
                gloss_dict_path=gloss_dict_path,
                ground_truth_path=ground_truth_path
            )
        except Exception as e:
            # 如果预处理出错，打印错误信息并返回
            print(f"预处理出错: {e}")
            return

        # 加载词汇表
        with open(os.path.join(gloss_dict_path, f'{dataset_name}_gloss_dict.npy'), 'rb') as f:
            gloss_dict = np.load(f, allow_pickle=True).item()

        # 根据配置初始化 Phoenix2014DataModule 对象
        datamodule = Phoenix2014DataModule(
            features_path=features_path,
            annotations_path=annotations_path,
            gloss_dict=gloss_dict,  # 词汇表，例如 {'gloss1': 0, 'gloss2': 1}
            **datamodule_cfg
        )
        return datamodule
    elif dataset_name == 'phoenix2014T':
        # TODO: 实现 Phoenix2014T 数据集的处理逻辑
        return None
    elif dataset_name == 'csl-daily':
        # TODO: 实现 CSL-Daily 数据集的处理逻辑
        return None
    else:
        # 如果提供的数据集名称不匹配任何已知数据集，抛出异常
        raise ValueError(f"不支持的数据集: {dataset_name}")
