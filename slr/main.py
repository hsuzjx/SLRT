import os
from datetime import datetime

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig

from slr.model import SLRModel
from .utils import *

CONFIG_PATH = '../configs'
CONFIG_NAME = 'CorrNet_experiment.yaml'


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    """
    主函数，用于执行整个训练流程。

    :param cfg: 包含所有配置信息的对象
    """
    # 配置设置
    torch.set_float32_matmul_precision(cfg.get('torch_float32_matmul_precision', 'high'))

    # 随机种子设置
    seed = set_seed(cfg.get('seed', -1), workers=True)

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
        project=project,
        name=f'{name}_{timestamp}',
        update_config={'random_seed': seed},
        logger_cfg=logger_cfg
    )

    # 检查点回调设置
    checkpoint_callback = init_checkpoint_callback(
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
    data_module = init_datamodule(
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
    model = init_model(
        save_dir=save_dir,
        dataset_name=dataset_name,
        gloss_dict=gloss_dict,
        ground_truth_path=ground_truth_dir,
        model_name=cfg.get('model_name'),
        model_cfg=model_cfg
    )

    # trainer
    trainer = init_trainer(
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
