import os
import random

import hydra
import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.data import Phoenix2014DataModule
from src.data import Phoenix2014TDataModule

from src.model import SLRModel
from src.utils import preprocess


# 安全地从配置中检索值，使用默认值作为回退。
def safe_get_config(cfg, section, default=None):
    """
    从配置对象中安全地获取指定部分的配置信息。

    :param cfg: 配置对象
    :param section: 要获取的配置部分
    :param default: 默认值，在无法找到对应配置时返回
    :return: 配置信息或默认值
    """
    if default is None:
        default = {}
    try:
        return OmegaConf.select(cfg, section, default=default)
    except KeyError:
        return default


# 设置随机种子，基于配置。
def setup_seed(cfg):
    """
    根据配置设置全局随机种子。

    :param cfg: 包含随机种子配置的对象
    """
    seed = cfg.get('seed', -1)
    if seed == -1:
        seed = random.randint(1, 2 ** 32 - 1)
    L.seed_everything(seed, workers=True)
    return seed


# 初始化Weights & Biases日志记录器。
def setup_wandb_logger(cfg, project, name, timestamp, update_config=None):
    """
    创建并初始化Weights & Biases日志记录器。

    :param cfg: 包含日志相关配置的对象
    :param project: 项目名称
    :param name: 运行名称
    :param timestamp: 时间戳
    :param update_config: 需要更新的配置
    :return: WandbLogger实例
    """
    save_dir = os.path.join(cfg.get('save_dir', '.'), f'{project}/{name}/{timestamp}')
    os.makedirs(save_dir, exist_ok=True)  # 确保不会因为目录已存在而抛出异常
    is_offline = cfg.get('offline', False)
    wandb_logger = WandbLogger(project=project, name=f"{name}_{timestamp}", offline=is_offline, save_dir=save_dir)
    wandb.require("core")

    if update_config:
        wandb_logger.experiment.config.update(update_config, allow_val_change=True)

    return wandb_logger


# 设置模型检查点回调。
def setup_checkpoint_callback(cfg, project, name, timestamp):
    """
    创建并初始化模型检查点回调。

    :param cfg: 包含检查点相关配置的对象
    :param project: 项目名称
    :param name: 运行名称
    :param timestamp: 时间戳
    :return: ModelCheckpoint实例
    """
    dirpath = os.path.join(cfg.get('save_dir', '.'), f'{project}/{name}/{timestamp}/checkpoints')
    os.makedirs(dirpath, exist_ok=True)  # 确保不会因为目录已存在而抛出异常
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor=cfg.get('monitor', 'val_loss'),
        mode=cfg.get('mode', 'min'),
        save_last=cfg.get('save_last', True),
        save_top_k=cfg.get('save_top_k', 1)
    )
    return checkpoint_callback


@hydra.main(version_base=None, config_path='../configs', config_name='main.yaml')
def main(cfg: DictConfig):
    """
    主函数，用于执行整个训练流程。

    :param cfg: 包含所有配置信息的对象
    """
    # 配置设置
    torch.set_float32_matmul_precision(cfg.get('torch_float32_matmul_precision', 'high'))

    # 随机种子设置
    seed = setup_seed(cfg)

    project = cfg.get('project', 'default_project')
    name = cfg.get('name', 'default_name')
    timestamp = cfg.get('timestamp', '00000000')

    # Wandb日志记录器设置
    wandb_cfg = safe_get_config(cfg, 'logger')
    wandb_logger = setup_wandb_logger(wandb_cfg, project, name, timestamp, {'random_seed': seed})

    # 检查点回调设置
    checkpoint_cfg = safe_get_config(cfg, 'callback')
    checkpoint_callback = setup_checkpoint_callback(checkpoint_cfg, project, name, timestamp)

    # 预处理设置
    preprocess_cfg = safe_get_config(cfg, 'preprocess')
    dataset_name = preprocess_cfg.get('name')
    try:
        preprocess(
            dataset_name=dataset_name,
            annotations_path=preprocess_cfg.get('annotations_path'),
            gloss_dict_path=preprocess_cfg.get('gloss_dict_path'),
            ground_truth_path=preprocess_cfg.get('ground_truth_path')
        )
    except Exception as e:
        print(f"预处理出错: {e}")
        return

    # 模型和数据模块设置
    train_cfg = safe_get_config(cfg, 'train')
    gloss_dict_path = preprocess_cfg.get('gloss_dict_path')
    if gloss_dict_path is None or not os.path.exists(gloss_dict_path):
        print("找不到词汇表路径。")
        return
    gloss_dict = np.load(os.path.join(gloss_dict_path, f'{dataset_name}_gloss_dict.npy'), allow_pickle=True).item()
    data_module = Phoenix2014DataModule(
        features_path=preprocess_cfg.get('features_path'),
        annotations_path=preprocess_cfg.get('annotations_path'),
        gloss_dict=gloss_dict,
        num_workers=train_cfg.get('num_workers'),
        batch_size=train_cfg.get('batch_size'),
    )
    model_cfg = safe_get_config(cfg, 'model')
    model = SLRModel(
        num_classes=model_cfg.num_classes, conv_type=2, use_bn=False, hidden_size=1024,
        gloss_dict=gloss_dict,
        weight_norm=True,
        lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
        last_epoch=-1,
        test_param=False,
        save_path=os.path.join(train_cfg.get('save_path', '.'),
                               f'{project}/{name}/{timestamp}/hypothesis'),
        # for evaluation
        dataset_name=dataset_name,
        evaluation_sh_path=train_cfg.get('sh_path'),
        ground_truth_path=preprocess_cfg.get('ground_truth_path'),
        evaluation_sclite_path=train_cfg.get('sclite_path'),
    )

    # 训练器设置
    trainer = L.Trainer(
        max_epochs=train_cfg.get('n_epochs'),
        accelerator=train_cfg.get('accelerator'),
        devices=train_cfg.get('devices'),
        precision=train_cfg.get('precision'),
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true'
    )

    try:
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
    except Exception as e:
        print(f"训练过程中出错: {e}")
        wandb.finish(exit_code=1)
    finally:
        wandb.finish()  # 确保资源被释放
        # 可以考虑在这里加入更多的清理工作，比如关闭文件句柄等


if __name__ == '__main__':
    main()
