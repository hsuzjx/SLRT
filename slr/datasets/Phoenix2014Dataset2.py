import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class Phoenix2014Dataset(data.Dataset):
    """
    A dataset class for handling the Phoenix-2014 dataset.

    Parameters:
    - dataset_dir (str, optional): Path to the main dataset directory.
    - features_path (str, optional): Path to the feature files.
    - annotations_path (str, optional): Path to the annotation files.
    - mode (str, optional): The dataset mode, must be one of "train", "dev", "test". Defaults to "train".
    - frame_size (tuple, optional): Size of video frames. Default is (210, 260).
    - drop_ids (list, optional): List of sample IDs to be dropped. Default is None.
    - .....
    - ....

    Returns:
    None
    """

    def __init__(
            self,
            dataset_dir: str = None,
            features_path: str = None,
            annotations_path: str = None,
            mode: str = "train",
            frame_size: tuple = (210, 260),
            drop_ids: list = None,
            transforms=None,
            tokenizer=None
    ) -> None:
        super().__init__()

        # Verify the correctness of the mode parameter, raise an exception if it is not one of 'train', 'dev', or 'test'
        if mode not in ["train", "dev", "test"]:
            raise ValueError("mode must be one of 'train', 'dev', or 'test'")
        self.mode = mode

        # Determine paths based on provided arguments
        if dataset_dir is not None:
            self.features_path = os.path.join(
                dataset_dir, f'phoenix-2014-multisigner/features/fullFrame-{frame_size[0]}x{frame_size[1]}px'
            )
            self.annotations_path = os.path.join(
                dataset_dir, 'phoenix-2014-multisigner/annotations/manual'
            )
        elif features_path is not None and annotations_path is not None:
            self.features_path = os.path.abspath(features_path)
            self.annotations_path = os.path.abspath(annotations_path)
        else:
            raise ValueError(
                "Either dataset_dir must be provided or both features_path and annotations_path must be provided.")

        # Check if paths are valid
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features directory not found at {self.features_path}")
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_path}")

        corpus_file = os.path.join(self.annotations_path, f'{self.mode}.corpus.csv')
        try:
            self.corpus = pd.read_csv(corpus_file, sep='|', header=0, index_col='id')
        except FileNotFoundError:
            raise FileNotFoundError(f"Corpus file not found at {corpus_file}")

        # Drop specific ID if needed
        if drop_ids is not None:
            for drop_id in drop_ids:
                if drop_id in self.corpus.index:
                    self.corpus.drop(drop_id, axis=0, inplace=True)
                else:
                    print(f"ID {drop_id} not found in corpus, skipping.")

        self.transforms = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.corpus)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset by index.

        Parameters:
        - idx (int): Index of the item to retrieve.

        Returns:
            - video (numpy.ndarray): Video frames as an array.
            - annotation (list): List of annotations.
            - info (pandas.Series): Item information.
        """

        # Retrieve the entry information at the specified index from the corpus DataFrame
        info = self.corpus.iloc[idx]

        # Find and sort frame files for the given folder
        frame_file_list = sorted(glob.glob(os.path.join(self.features_path, f'{self.mode}', info.folder)))
        if not frame_file_list:
            raise ValueError(f"No images found for folder {info.folder}")

        # Load and process each frame
        frames = []
        for frame_file in frame_file_list:
            frame = cv2.imread(frame_file)
            if frame is None:
                raise FileNotFoundError(f"Image not found or failed to load: {frame_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Convert frames to a NumPy array
        video = np.array(frames).astype(np.float32)

        # Normalize the video array
        video = video / 127.5 - 1

        # Split and clean the annotation string
        annotation = info.annotation.split(' ')
        annotation = [word for word in annotation if word]  # Remove empty strings

        if self.transforms:
            video = self.transforms(video)
        if self.tokenizer:
            annotation = self.tokenizer(annotation)

        return video, annotation, info

    @staticmethod
    def collate_fn(batch):
        """
        """
        # 按视频长度降序排序批次数据
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        # 解压缩batch，分别获取视频数据、标签数据和额外信息
        video, label, info = list(zip(*batch))
        # 对视频数据进行padding处理
        if len(video[0].shape) > 3:
            # 找到批次中最长的视频长度
            max_len = len(video[0])
            # 计算每个视频的padding后长度
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            # 左侧padding大小
            left_pad = 6
            # 右侧padding大小
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            # 更新最大长度以包括padding
            max_len = max_len + left_pad + right_pad
            # 对每个视频进行padding
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            # 将padding后的视频数据堆叠成张量
            padded_video = torch.stack(padded_video)
        else:
            # 找到批次中最长的视频长度
            max_len = len(video[0])
            # 计算每个视频的原始长度
            video_length = torch.LongTensor([len(vid) for vid in video])
            # 对每个视频进行padding
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            # 将padding后的视频数据堆叠成张量并转换维度
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        # 计算每个标签的长度
        label_length = torch.LongTensor([len(lab) for lab in label])
        # 根据是否有标签数据进行处理
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            # 对标签数据进行padding
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info
