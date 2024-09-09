import lightning as L
from omegaconf import DictConfig


def setup_trainer(logger, callbacks, trainer_cfg: DictConfig):
    """
    设置训练器配置。

    :param logger: 日志记录器，用于记录训练过程的信息
    :param callbacks: 回调函数列表，用于在训练过程中执行特定的操作
    :param trainer_cfg: 训练器配置字典，包含训练的各种设置
    :return: 配置好的 Trainer 对象，用于执行训练过程
    """

    # 根据配置文件创建 Trainer 实例
    trainer = L.Trainer(
        logger=logger,  # 配置日志记录器
        callbacks=callbacks,  # 配置回调函数
        **trainer_cfg
    )

    return trainer
