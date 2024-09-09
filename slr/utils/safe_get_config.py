from omegaconf import OmegaConf


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
