import numbers
import random

import PIL
import numpy as np


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        if isinstance(clip[0], np.ndarray):
            return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]
