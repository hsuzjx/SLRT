import random

import lightning as L


def set_seed(seed, workers=True):
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
