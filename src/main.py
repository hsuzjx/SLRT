import os
import random
import time

import hydra
import numpy as np
import torch
import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
from src.data import Phoenix2014DataModule
from src.model import SLRModel
from src.utils import preprocess


# 导入必要的库和模块，根据实际情况可能需要添加更多安全和异常处理相关的库

def safe_get_config(cfg, section, default=None):
    """安全地从配置中检索值，使用默认值作为回退。"""
    if default is None:
        default = {}
    try:
        return OmegaConf.select(cfg, section, default=default)
    except KeyError:
        return default


def setup_seed(cfg):
    """设置随机种子，基于配置。"""
    seed = cfg.get('seed', -1)
    if seed == -1:
        seed = random.randint(1, 2 ** 32 - 1)
    cfg.seed = seed
    L.seed_everything(seed, workers=True)


def setup_wandb_logger(cfg, project, name):
    """初始化Weights & Biases日志记录器。"""
    save_dir = os.path.join(cfg.get('save_dir', '.'), f'{project}/{name}')
    os.makedirs(save_dir, exist_ok=True)
    is_offline = cfg.get('offline', False)
    wandb_logger = WandbLogger(project=project, name=name, offline=is_offline, save_dir=save_dir)
    wandb.require("core")
    return wandb_logger


def setup_checkpoint_callback(cfg, project, name):
    """设置模型检查点回调。"""
    dirpath = os.path.join(cfg.get('save_dir', '.'), f'{project}/{name}/checkpoints')
    os.makedirs(dirpath, exist_ok=True)
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
    # 配置设置
    torch.set_float32_matmul_precision(cfg.get('torch_float32_matmul_precision', 'high'))

    # 随机种子设置
    setup_seed(cfg)

    project = cfg.get('project', 'default_project')
    name_cfg = safe_get_config(cfg, 'name')
    name = name_cfg.get('display_name', 'default_name')
    if name_cfg.get('include_time', False):
        name = f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"

    # Wandb日志记录器设置
    wandb_cfg = safe_get_config(cfg, 'logger')
    wandb_logger = setup_wandb_logger(wandb_cfg, project, name)

    # 检查点回调设置
    checkpoint_cfg = safe_get_config(cfg, 'callback')
    checkpoint_callback = setup_checkpoint_callback(checkpoint_cfg, project, name)

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
    model = SLRModel(
        num_classes=1296, conv_type=2, use_bn=False, hidden_size=1024,
        gloss_dict=gloss_dict,
        save_path=os.path.join(train_cfg.get('save_path', '.'), f'{project}/{name}/hypothesis'),
        sh_path=train_cfg.get('sh_path'),
        ground_truth_path=preprocess_cfg.get('ground_truth_path'),
        mer_path=train_cfg.get('mer_path'),
        weight_norm=True,
        lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
        last_epoch=-1,
        test_param=False,
    )

    # 训练器设置
    # trainer_cfg = safe_get_config(cfg, 'trainer')
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
        wandb.finish()


if __name__ == '__main__':
    main()
