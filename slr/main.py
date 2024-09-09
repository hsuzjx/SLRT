import os
import random
from datetime import datetime

import hydra
import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

import slr
from slr.data import Phoenix2014DataModule
from slr.data import Phoenix2014TDataModule
from slr.model import SLRModel
from .data.preprocess import preprocess

CONFIG_PATH = '../configs'
CONFIG_NAME = 'CorrNet_experiment.yaml'


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
        save_dir=save_dir,
        **logger_cfg
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
        **callback_cfg
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


def setup_model(
        save_dir, dataset_name, gloss_dict, ground_truth_path, model_name, model_cfg: DictConfig
):
    """
    根据配置设置和初始化模型。
    
    :param model_name: 字符串，表示模型的名称。
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
    try:
        model = getattr(slr.model, model_name)(
            save_path=save_path,  # 模型保存路径
            dataset_name=dataset_name,  # 数据集名称
            gloss_dict=gloss_dict,  # 标签字典
            ground_truth_path=ground_truth_path,  # 真实标签路径
            **model_cfg
        )
    except AttributeError as e:
        raise ValueError(f"Model '{model_name}' not found in src.model.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the model with name '{model_name}'.") from e
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
        logger=logger,  # 配置日志记录器
        callbacks=callbacks,  # 配置回调函数
        **trainer_cfg
    )

    return trainer


def convert_to_onnx(model, file_path):
    """
    将给定的模型转换为ONNX格式并保存到指定的文件路径。

    参数:
    - model: 需要转换的PyTorch模型。
    - file_path: 保存转换后的ONNX模型的文件路径。

    返回值:
    无
    """
    try:
        # 将模型移动到CPU
        model = model.to('cpu')

        # 设置模型为评估模式
        model.eval()

        # 定义输入样例。这里的样例包括两个输入张量和对应的标签张量。
        input_sample = (torch.randn(2, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100, 100]).to('cpu'))

        # 导出模型到ONNX格式。export_params=True表示将模型的参数一起导出。
        model.to_onnx(file_path, input_sample, export_params=True)

    except Exception as e:
        # 捕获并打印可能发生的异常
        print(f"Error occurred: {e}")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
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
    timestamp = cfg.get('timestamp', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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
        dataset_name=dataset_name,
        gloss_dict=gloss_dict,
        ground_truth_path=ground_truth_dir,
        model_name=cfg.get('model_name'),
        model_cfg=model_cfg
    )

    # trainer
    trainer = setup_trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        trainer_cfg=trainer_cfg
    )

    # train model
    trainer.fit(model, datamodule=data_module)

    # test the best model
    best_model = SLRModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    trainer.test(best_model, datamodule=data_module)

    # 确保wandb.finish()被执行，以释放资源
    try:
        wandb.finish()
    except Exception as finish_error:
        print(f"wandb.finish() 出现问题: {finish_error}")

    # 根据配置决定是否将模型转换为ONNX格式
    if cfg.get('convert_to_onnx', False):
        # 加载最佳模型以进行ONNX转换
        best_model = SLRModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        # 创建保存ONNX模型的目录
        onnx_save_dir = os.path.join(save_dir, 'onnx')
        os.makedirs(onnx_save_dir, exist_ok=True)
        # 执行模型到ONNX的转换
        convert_to_onnx(best_model, os.path.join(onnx_save_dir, 'best_model.onnx'))


if __name__ == '__main__':
    main()
