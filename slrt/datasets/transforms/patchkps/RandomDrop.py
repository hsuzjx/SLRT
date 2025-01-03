import random


class RandomDrop(object):
    def __init__(self, drop_ratio=0.1):
        """
        初始化方法。

        :param drop_ratio: 随机丢弃 T 维度的比例，默认为 0.1
        """
        super(RandomDrop, self).__init__()
        self.drop_ratio = drop_ratio

    def __call__(self, patches, kps):
        """
        前向传播方法，对输入张量进行处理。

        :param x: 输入张量，形状为 (T,V,C)
        :return: 处理后的张量，形状为 (T',V,C)，其中 T' 是 4 的倍数
        """
        T, V, C = kps.shape

        # 计算需要保留的 T 维度的数量
        num_to_keep = max(1, int(T * (1 - self.drop_ratio)))

        # 确保 num_to_keep 是 4 的倍数
        num_to_keep = (num_to_keep // 4) * 4

        # 随机选择要保留的索引
        indices = random.sample(range(T), num_to_keep)
        indices.sort()

        return patches[indices, :, :, :, :], kps[indices, :, :]
