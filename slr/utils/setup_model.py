import os

import slr


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
