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
    # 初始化默认值为一个空字典，以便在无法找到配置时提供一个空的字典作为返回值
    if default is None:
        default = {}
    try:
        # 尝试从配置对象中获取指定部分的配置信息，如果存在则返回相应的配置信息
        return OmegaConf.select(cfg, section, default=default)
    except KeyError:
        # 如果指定部分的配置不存在，则返回默认值
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


def init_wandb_logger(save_dir, wandb_project, wandb_name, logger_cfg: DictConfig, update_config=None):
    """
    创建并初始化Weights & Biases日志记录器。

    :param save_dir: 日志保存的目录
    :param wandb_project: 项目名称
    :param wandb_name: 运行名称
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
        project=wandb_project,
        name=wandb_name,
        offline=logger_cfg.get('offline', False),
        save_dir=save_dir
    )

    # 如果有配置更新，更新WandbLogger的配置
    if isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run) and update_config:
        wandb_logger.experiment.config.update(update_config, allow_val_change=True)

    # 返回初始化的日志记录器
    return wandb_logger


def setup_checkpoint_callback(save_dir, callback_cfg: DictConfig):
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
        monitor=callback_cfg.get('monitor', 'DEV_WER'),
        mode=callback_cfg.get('mode', 'min'),
        save_last=callback_cfg.get('save_last', True),
        save_top_k=callback_cfg.get('save_top_k', 1)
    )
    # 返回ModelCheckpoint实例
    return checkpoint_callback


def setup_datamodule(dataset_name, features_path, annotations_path, gloss_dict_path, ground_truth_path,
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
            num_workers=datamodule_cfg.get('num_workers', 8),  # 获取配置中的工作线程数，默认为 8
            batch_size=datamodule_cfg.get('batch_size', 2)  # 获取配置中的批次大小，默认为 2
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


def setup_model(save_dir, gloss_dict, dataset_name, ground_truth_path, model_cfg: DictConfig):
    """
    根据配置设置和初始化模型。

    :param save_dir: 字符串，表示模型保存的目录。
    :param gloss_dict: 字典，表示词汇表。
    :param dataset_name: 字符串，表示数据集的名称。
    :param ground_truth_path: 字符串，表示地面真实文件的路径。
    :param model_cfg: DictConfig 对象，包含模型的配置。
    :return: 返回初始化后的模型对象。
    """

    # 创建保存路径
    save_path = os.path.join(save_dir, 'hypothesis')
    os.makedirs(save_path, exist_ok=True)

    # 构造模型
    model = SLRModel(
        # for common
        save_path=save_path,  # wer保存路径
        test_param=model_cfg.get('test_param', False),  # 是否测试参数

        # for network
        num_classes=model_cfg.get('num_classes', -1),  # 分类数
        conv_type=model_cfg.get('conv_type', 2),  # 卷积类型
        use_bn=model_cfg.get('use_bn', False),  # 不使用批量归一化
        hidden_size=model_cfg.get('hidden_size', 1024),  # 隐藏层大小

        # for decoder
        gloss_dict=gloss_dict,  # 标签字典

        # for optimizer and lr_scheduler
        lr=model_cfg.get('lr', 0.0001),  # 学习率，默认值0.0001
        weight_norm=model_cfg.get('weight_norm', True),  # 是否使用权重归一化
        weight_decay=model_cfg.get('weight_decay', 0.0001),  # 权重衰减，默认值0.0001
        lr_scheduler_milestones=model_cfg.get('lr_scheduler_milestones', None),  # 学习率调度的里程碑，默认为空列表
        lr_scheduler_gamma=model_cfg.get('lr_scheduler_gamma', 0.2),  # 学习率调度的因子
        last_epoch=model_cfg.get('last_epoch', -1),  # 上一个训练周期

        # for evaluation
        dataset_name=dataset_name,  # 数据集名称
        evaluation_sh_path=os.path.abspath(model_cfg.get('evaluation_sh_path')),  # 评估脚本路径
        ground_truth_path=ground_truth_path,  # 真实标签路径
        evaluation_sclite_path=os.path.abspath(model_cfg.get('evaluation_sclite_path')),  # sclite评估工具路径
    )
    return model


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
        max_epochs=trainer_cfg.get('max_epochs', 1),  # 最大训练周期数
        accelerator=trainer_cfg.get('accelerator', 'cpu'),  # 训练所使用的加速器类型
        devices=trainer_cfg.get('devices', 1),  # 使用的设备数量
        precision=trainer_cfg.get('precision', 32),  # 训练的精度，如32位或16位
        logger=logger,  # 配置日志记录器
        callbacks=callbacks,  # 配置回调函数
        strategy=trainer_cfg.get('strategy', 'ddp_find_unused_parameters_true'),
        # 分布式训练策略，默认为 'ddp_find_unused_parameters_true'
        limit_train_batches=trainer_cfg.get('limit_train_batches', 1.0),  # 训练批次的数据使用比例
        limit_val_batches=trainer_cfg.get('limit_val_batches', 1.0),  # 验证批次的数据使用比例
        limit_test_batches=trainer_cfg.get('limit_test_batches', 1.0),  # 测试批次的数据使用比例
    )

    return trainer


@hydra.main(version_base=None, config_path='../configs', config_name='example1.yaml')
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
    # # 初始化分布式进程组
    # if 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     local_rank = int(os.environ['LOCAL_RANK'])
    # else:
    #     rank, world_size, local_rank = 0, 1, 0
    #
    # if rank==0:
    #     # timestamp = cfg.get('timestamp', '00000000')
    #     timestamp = datetime.now().timestamp()
    #     timestamp_tensor = torch.tensor(timestamp)
    # else:
    #     timestamp_tensor = torch.tensor(0.0)
    # torch.distributed.broadcast(timestamp_tensor, src=0)
    # torch.distributed.barrier()
    #
    # # 更新时间戳
    # timestamp = timestamp_tensor.item()
    timestamp = cfg.get('timestamp', '00000000')

    # 创建保存目录
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 获取配置
    data_cfg = safe_get_config(cfg, 'data')
    datamodule_cfg = safe_get_config(cfg, 'datamodule')
    logger_cfg = safe_get_config(cfg, 'logger')
    callback_cfg = safe_get_config(cfg, 'callback')
    model_cfg = safe_get_config(cfg, 'model')
    trainer_cfg = safe_get_config(cfg, 'trainer')

    # Wandb日志记录器设置
    # TODO: 修改参数名字，使其无歧义
    wandb_logger = init_wandb_logger(
        save_dir=save_dir,
        wandb_project=project,
        wandb_name=f'{name}_{timestamp}',
        update_config={'random_seed': seed},
        logger_cfg=logger_cfg
    )

    # 检查点回调设置
    checkpoint_callback = setup_checkpoint_callback(
        save_dir=save_dir,
        callback_cfg=callback_cfg
    )

    # 共用参数
    dataset_name = data_cfg.get('name')
    features_dir = os.path.abspath(data_cfg.get('features_dir'))
    annotations_dir = os.path.abspath(data_cfg.get('annotations_dir'))

    gloss_dict_dir = os.path.abspath(data_cfg.get('gloss_dict_dir'))
    ground_truth_dir = os.path.abspath(data_cfg.get('ground_truth_dir'))
    os.makedirs(gloss_dict_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)

    # 数据模块初始化
    data_module = setup_datamodule(
        dataset_name=dataset_name,
        features_path=features_dir,
        annotations_path=annotations_dir,
        gloss_dict_path=gloss_dict_dir,
        ground_truth_path=ground_truth_dir,
        datamodule_cfg=datamodule_cfg
    )

    with open(os.path.join(gloss_dict_dir, f'{dataset_name}_gloss_dict.npy'), 'rb') as f:
        gloss_dict = np.load(f, allow_pickle=True).item()

    # 模型初始化
    model = setup_model(
        save_dir=save_dir,
        gloss_dict=gloss_dict,
        dataset_name=dataset_name,
        ground_truth_path=ground_truth_dir,
        model_cfg=model_cfg
    )

    # trainer
    trainer = setup_trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        trainer_cfg=trainer_cfg
    )

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
    # TODO: **kwargs参数形式的函数
    main()
