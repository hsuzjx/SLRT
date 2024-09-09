import os

from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig


def init_checkpoint_callback(save_dir, callback_cfg: DictConfig):
    """
    创建并初始化模型检查点回调。

    :param save_dir: 检查点文件保存的目录路径
    :param callback_cfg: 回调配置的字典，包含监控量、模式、是否保存最后一个epoch的模型和保存top k个最好的模型等信息
    :return: ModelCheckpoint实例
    """
    # 构建检查点保存的目录路径
    dirpath = os.path.join(save_dir, 'checkpoints')
    # 确保不会因为目录已存在而抛出异常
    os.makedirs(dirpath, exist_ok=True)
    # 初始化ModelCheckpoint实例
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        **callback_cfg
    )
    # 返回ModelCheckpoint实例
    return checkpoint_callback
