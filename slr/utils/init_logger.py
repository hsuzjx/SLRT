import os

import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig


def init_wandb_logger(save_dir, project, name, logger_cfg: DictConfig, update_config=None):
    """
    创建并初始化Weights & Biases日志记录器。

    :param save_dir: 日志保存的目录
    :param project: 项目名称
    :param name: 运行名称
    :param logger_cfg: DictConfig类型的日志配置
    :param update_config: 需要更新的配置（可选）
    :return: WandbLogger实例，用于在整个训练过程中记录日志
    """
    # 确保W&B库的核心功能可用
    wandb.require("core")
    # 创建保存目录，如果目录不存在
    os.makedirs(save_dir, exist_ok=True)

    # 初始化WandbLogger实例
    wandb_logger = WandbLogger(
        project=project,
        name=name,
        save_dir=save_dir,
        **logger_cfg
    )

    # 如果有配置更新，更新WandbLogger的配置
    if isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run) and update_config:
        wandb_logger.experiment.config.update(update_config, allow_val_change=True)

    # 返回初始化的日志记录器
    return wandb_logger
