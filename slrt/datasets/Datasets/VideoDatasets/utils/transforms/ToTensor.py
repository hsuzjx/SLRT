import numpy as np
import torch


class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()  # (T, H, W, C) -> (T, C, H, W)
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))  # (T, H, W, C) -> (T, C, H, W)
        return video
