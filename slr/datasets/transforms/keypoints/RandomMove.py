import numpy as np
import torch


class RandomMove(object):
    """
    Randomly move the keypoints in the image.
    """

    def __init__(self):
        """
        初始化方法。

        :param drop_ratio: 随机丢弃 T 维度的比例，默认为 0.1
        """
        super(RandomMove, self).__init__()

    def __call__(self, x):
        """
        输入和输出都是 torch.Tensor 类型。
        """
        # 将输入张量转换为 numpy 数组
        data_numpy = x[:2, :, :].permute(1, 2, 0).numpy()

        # input: C,T,V,M
        # C, T, V, M = data_numpy.shape
        degrees = np.random.uniform(-15, 15)
        theta = np.radians(degrees)
        p = np.random.uniform(0, 1)
        if p >= 0.5:
            data_numpy = self.rotate_points(data_numpy, theta)
        # dx = np.random.uniform(-0.21, 0.21)
        # dy = np.random.uniform(-0.26, 0.26)
        # data_numpy = self.translate_points(data_numpy, [dx, dy])
        # scale = np.random.uniform(0.8, 1.2)
        # p = np.random.uniform(0, 1)
        # if p >= 0.5:
        #     data_numpy = self.scale_points(data_numpy, scale)

        x[:2, :, :] = torch.from_numpy(data_numpy).permute(2, 0, 1)
        return x

    def rotate_points(self, points, angle):
        center = [0, 0]

        # 将坐标平移到原点
        points_centered = points - center

        # 构建旋转矩阵，注意方向
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        # 进行坐标点旋转
        points_rotated = np.dot(points_centered, rotation_matrix.T)

        # 将坐标平移到原来的中心位置
        points_transformed = points_rotated + center

        return points_transformed

    def translate_points(self, points, translation):
        # 进行平移操作
        points_translated = points + translation

        return points_translated

    def scale_points(self, points, scale_factor):
        # 缩放坐标
        points_scaled = points * scale_factor

        return points_scaled
