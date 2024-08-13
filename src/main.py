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


def setup_seed(seed, workers=True):
    """
    根据配置设置全局随机种子。

    :param seed: 随机种子值，如果为-1，则会随机生成一个种子值
    :param workers: 是否为不同workers设置不同种子，默认为True
    :return: 设置的随机种子值
    """
    # 当seed值为-1时，生成一个随机的种子值
    if seed == -1:
        seed = random.randint(1, 2 ** 32 - 1)
    # 使用生成或传入的种子值设置全局随机种子
    L.seed_everything(seed, workers=workers)
    return seed


def init_wandb_logger(save_dir, project, name, timestamp, update_config=None, is_offline=False):
    """
    创建并初始化Weights & Biases日志记录器。

    :param save_dir: 日志保存的目录
    :param project: 项目名称
    :param name: 运行名称
    :param timestamp: 时间戳，用于区分不同的运行
    :param update_config: 需要更新的配置（可选）
    :param is_offline: 是否离线模式，不上传数据到W&B服务器（可选）
    :return: WandbLogger实例，用于在整个训练过程中记录日志
    """
    # 确保W&B库的核心功能可用
    wandb.require("core")
    # 创建保存目录，如果目录不存在
    os.makedirs(save_dir, exist_ok=True)
    # 初始化WandbLogger实例
    wandb_logger = WandbLogger(project=project, name=f"{name}_{timestamp}", offline=is_offline, save_dir=save_dir)

    # 如果有配置更新，更新WandbLogger的配置
    if update_config:
        wandb_logger.experiment.config.update(update_config, allow_val_change=True)

    # 返回初始化的日志记录器
    return wandb_logger


def setup_checkpoint_callback(dirpath, monitor, mode, save_last, save_top_k):
    """
    创建并初始化模型检查点回调。

    :param dirpath: 检查点文件保存的目录路径
    :param monitor: 用于检查点保存的量，如验证集准确率
    :param mode: 监控量的模式，如'min'或'max'
    :param save_last: 是否保存最后一个epoch的模型
    :param save_top_k: 是否保存top k个最好的模型
    :return: ModelCheckpoint实例
    """
    # 确保不会因为目录已存在而抛出异常
    os.makedirs(dirpath, exist_ok=True)
    # 初始化ModelCheckpoint实例
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor,
        mode=mode,
        save_last=save_last,
        save_top_k=save_top_k
    )
    # 返回ModelCheckpoint实例
    return checkpoint_callback


def setup_datamodule(dataset_name, features_path, annotations_path, gloss_dict_path, ground_truth_path, num_workers,
                     batch_size):
    """
    根据配置设置数据模块。

    该函数根据预处理配置和训练配置初始化一个Phoenix2014DataModule数据模块，
    它负责加载和处理训练数据。

    参数:
    - preprocess_cfg: 预处理配置字典，包含数据路径等信息。
    - train_cfg: 训练配置字典，包含批处理大小和工作进程数等信息。
    - gloss_dict: 一个字典，映射手势标签到数字，用于数据的标签处理。

    返回:
    - Phoenix2014DataModule: 初始化后的数据模块对象，可以用于训练过程中的数据加载。
    """
    try:
        preprocess(
            dataset_name=dataset_name,
            annotations_path=annotations_path,
            gloss_dict_path=gloss_dict_path,
            ground_truth_path=ground_truth_path
        )
    except Exception as e:
        print(f"预处理出错: {e}")
        return

    with open(os.path.join(gloss_dict_path, f'{dataset_name}_gloss_dict.npy'), 'rb') as f:
        gloss_dict = np.load(f, allow_pickle=True).item()

    # TODO: Phoenix2014T and CSL-Daily
    # 初始化Phoenix2014DataModule对象，使用配置文件中提供的路径和参数
    datamodule = Phoenix2014DataModule(
        features_path=features_path,
        annotations_path=annotations_path,
        gloss_dict=gloss_dict,  # 确保 gloss_dict 是一个字典，例如 {'gloss1': 0, 'gloss2': 1}
        num_workers=num_workers,
        batch_size=batch_size,
    )
    return datamodule, gloss_dict


def setup_model(cfg: DictConfig):
    """
    根据配置文件设置和初始化模型。

    该函数从配置文件中提取信息，构建特定的模型实例，并设置模型的训练和评估参数。

    参数:
    - cfg: DictConfig类型，包含模型、训练和预处理的配置信息。

    返回:
    - 返回构建好的模型实例。
    """
    # 从配置文件中提取必要的参数
    model_cfg = cfg.model
    train_cfg = cfg.train
    preprocess_cfg = cfg.preprocess

    # 检查必要的配置项是否存在
    required_keys = ['num_classes', 'gloss_dict']
    if not all(key in model_cfg for key in required_keys):
        raise ValueError("Missing required keys in model configuration.")

    # 拼接保存路径
    save_path = os.path.join(train_cfg.get('save_path', '.'),
                             f'{cfg.project}/{cfg.name}/{cfg.timestamp}/hypothesis')

    # 构造模型
    # TODO: 分离 model，train，evaluation 参数
    model = SLRModel(
        # 
        num_classes=model_cfg.num_classes,  # 分类数
        conv_type=2,  # 卷积类型
        use_bn=False,  # 不使用批量归一化
        hidden_size=1024,  # 隐藏层大小

        # 
        gloss_dict=model_cfg.gloss_dict,  # 标签字典

        # 
        weight_norm=True,  # 使用权重归一化
        lr=model_cfg.get('lr', 0.0001),  # 学习率，默认值0.0001
        weight_decay=model_cfg.get('weight_decay', 0.0001),  # 权重衰减，默认值0.0001
        lr_scheduler_milestones=model_cfg.get('lr_scheduler_milestones', []),  # 学习率调度的里程碑，默认为空列表
        lr_scheduler_gamma=model_cfg.get('lr_scheduler_gamma', 0.2),  # 学习率调度的因子

        #
        last_epoch=model_cfg.get('last_epoch', -1),  # 上一个训练周期
        test_param=model_cfg.get('test_param', False),  # 测试参数

        # 
        save_path=save_path,  # 模型保存路径
        dataset_name=cfg.dataset_name,  # 数据集名称
        evaluation_sh_path=train_cfg.get('sh_path'),  # 评估脚本路径
        ground_truth_path=preprocess_cfg.get('ground_truth_path'),  # 真实标签路径
        evaluation_sclite_path=train_cfg.get('sclite_path'),  # sclite评估工具路径
    )
    return model


def setup_trainer(train_cfg, wandb_logger, checkpoint_callback):
    """
    设置训练器配置。

    :param train_cfg: 训练配置字典
    :param wandb_logger: WandB 日志记录器
    :param checkpoint_callback: 检查点回调函数
    :return: 配置好的 Trainer 对象
    """
    # 使用默认值防止 KeyError
    max_epochs = train_cfg.get('n_epochs', 1)
    accelerator = train_cfg.get('accelerator', 'cpu')
    devices = train_cfg.get('devices', 1)
    precision = train_cfg.get('precision', 32)

    # 创建 Trainer 实例
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        strategy=train_cfg.get('strategy', 'ddp_find_unused_parameters_true'),  # 提供默认策略
        limit_train_batches=train_cfg.get('limit_train_batches', 1.0)  # 允许配置训练批次限制
    )

    return trainer


@hydra.main(version_base=None, config_path='../configs', config_name='main.yaml')
def main(cfg: DictConfig):
    """
    主函数，用于执行整个训练流程。

    :param cfg: 包含所有配置信息的对象
    """
    # 配置设置
    torch.set_float32_matmul_precision(cfg.get('torch_float32_matmul_precision', 'high'))

    # 随机种子设置
    seed = cfg.get('seed', -1)
    seed = setup_seed(seed, workers=True)

    # 获取项目名称、名称和时间戳
    project = cfg.get('project', 'default_project')
    name = cfg.get('name', 'default_name')
    timestamp = cfg.get('timestamp', '00000000')

    # 创建保存目录
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 获取配置
    # TODO: 将 model，train 的配置，改成 model，train，evaluation，trainer。
    data_cfg = safe_get_config(cfg, 'data')
    logger_cfg = safe_get_config(cfg, 'logger')
    callback_cfg = safe_get_config(cfg, 'callback')
    model_cfg = safe_get_config(cfg, 'model')
    train_cfg = safe_get_config(cfg, 'train')


    # Wandb日志记录器设置
    wandb_logger = init_wandb_logger(save_dir=save_dir,
                                     project=project, name=name, timestamp=timestamp,
                                     update_config={'random_seed': seed},
                                     is_offline=logger_cfg.get('is_offline', False))

    # 检查点回调设置
    checkpoint_callback = setup_checkpoint_callback(dirpath=os.path.join(save_dir, 'checkpoints'),
                                                    monitor=callback_cfg.get('monitor', 'DEV_WER'),
                                                    mode=callback_cfg.get('mode', 'min'),
                                                    save_last=callback_cfg.get('save_last', True),
                                                    save_top_k=callback_cfg.get('save_top_k', 1))

    # 数据模块初始化
    data_module, gloss_dict = setup_datamodule(dataset_name=data_cfg.get('name', 'err_dataset_name'),
                                               features_path=os.path.abspath(
                                                   data_cfg.get('features_path', 'err_features_path')),
                                               annotations_path=os.path.abspath(
                                                   data_cfg.get('annotations_path', 'err_annotations_path')),
                                               gloss_dict_path=os.path.abspath(
                                                   data_cfg.get('gloss_dict_path', 'err_gloss_dict_path')),
                                               ground_truth_path=os.path.abspath(
                                                   data_cfg.get('ground_truth_path', 'err_ground_truth_path')),
                                               num_workers=data_cfg.get('num_workers', 0),
                                               batch_size=data_cfg.get('batch_size', 1))

    # 模型初始化
    model = setup_model(model_cfg)
    trainer = setup_trainer(train_cfg, wandb_logger, checkpoint_callback)

    # 异常处理
    try:
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
    except Exception as e:
        print(f"训练过程中出错: {e}")
        wandb.finish(exit_code=1)
    finally:
        try:
            wandb.finish()  # 确保资源被释放
        except Exception as finish_error:
            print(f"wandb.finish() 出现问题: {finish_error}")


if __name__ == '__main__':
    # TODO: 修改核查配置文件
    # TODO: 函数参数改为输入配置cfg，以适应不同配置文件
    main()
