import numpy as np


class SelectIndex(object):
    def __init__(self, tmin, tmax, clip_len):
        """
        """
        super(SelectIndex, self).__init__()
        self.tmin = tmin
        self.tmax = tmax
        self.clip_len = clip_len

    def __call__(self, patches, kps):
        vlen = kps.shape[0]  # kps (T,V,3)

        # 当时间范围限制为1时，尝试选择整个视频或视频的中心部分作为帧序列
        if self.tmin == 1 and self.tmax == 1:
            if vlen <= self.clip_len:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                sequence = np.arange(vlen)
                an = (vlen - self.clip_len) // 2
                en = vlen - self.clip_len - an
                frame_index = sequence[an: -en]
                valid_len = self.clip_len

            # 确保有效长度是4的倍数
            if (valid_len % 4) != 0:
                valid_len -= (valid_len % 4)
                frame_index = frame_index[:valid_len]

            # 断言确保帧序列长度与有效长度匹配
            assert len(frame_index) == valid_len, (frame_index, valid_len)
            return patches[frame_index, :, :, :, :], kps[frame_index, :, :]

        # 计算基于时间范围的最小和最大长度限制
        min_len = min(int(self.tmin * vlen), self.clip_len)
        max_len = min(self.clip_len, int(self.tmax * vlen))
        # 在最小和最大长度之间随机选择一个长度，并调整为4的倍数
        selected_len = np.random.randint(min_len, max_len + 1)
        if (selected_len % 4) != 0:
            selected_len += (4 - (selected_len % 4))

        # 根据选定的长度，随机选择帧序列索引
        if selected_len <= vlen:
            selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
        else:
            copied_index = np.random.randint(0, vlen, selected_len - vlen)
            selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

        # 根据选定长度和剪辑长度限制，确定最终的帧序列和有效长度
        if selected_len <= self.clip_len:
            frame_index = selected_index
            valid_len = selected_len
        else:
            assert False, (vlen, selected_len, min_len, max_len)

        # 再次断言确保帧序列长度与有效长度匹配
        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return patches[frame_index, :, :, :, :], kps[frame_index, :, :]
