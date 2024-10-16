import argparse
import glob
import os
import pickle

import cv2
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from tqdm import tqdm


class CSLDailyPreprocesser:
    def __init__(self, dataset_dir, features_output_dir, features_dir=None, annotations_dir=None):
        # Ensure all directory paths are set correctly
        self.features_dir = os.path.join(dataset_dir, 'sentence_frames-512x512/frames_512x512') \
            if features_dir is None and dataset_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(dataset_dir, 'sentence_label') \
            if annotations_dir is None and dataset_dir is not None else os.path.abspath(
            annotations_dir) if annotations_dir else None
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        self.output_dir = features_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.transform = Compose([
            ToTensor(),
            Resize((256, 256)),
            Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        ])

    def preprocess(self):
        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info'])

        for item in tqdm(info['name'], total=len(info), desc="Preprocessing"):
            self._preprocess_item(item)

    def _preprocess_item(self, item):
        frames_dir = os.path.join(self.features_dir, item)
        video_output_dir = os.path.join(self.output_dir, item)
        os.makedirs(video_output_dir, exist_ok=True)

        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found at {frames_dir}")
        frames = self._read_frames(frames_dir)

        frames_transformed = []
        for frame in frames:
            frames_transformed.append(self.transform(frame))

        video = torch.stack(frames_transformed)

        torch.save(video.to(torch.float16), os.path.join(video_output_dir, "video.pt"))

    def _read_frames(self, frames_dir):
        """
        Reads video frames from the specified directory.

        Args:
            frames_dir (str): Directory containing video frames.

        Returns:
            list: List of frames read from the directory.
        """
        frames_file_list = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
        frames = []
        for frame_file in frames_file_list:
            frame = cv2.imread(frame_file)
            if frame is None:
                raise ValueError(f"Failed to read frame from {frame_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames


# 当脚本被直接运行时，执行以下代码
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="")

    # 添加命令行参数
    parser.add_argument("--dataset_dir", type=str, help="", default="../csl-daily")
    parser.add_argument("--features_output_dir", type=str, help="",
                        default="./csl-daily_features_transformed")
    parser.add_argument("--features_dir", type=str, help="", default=None)
    parser.add_argument("--annotations_dir", type=str, help="", default=None)

    # 解析命令行参数
    args = parser.parse_args()

    p = CSLDailyPreprocesser(
        dataset_dir=args.dataset_dir,
        features_output_dir=args.features_output_dir,
        features_dir=args.features_dir,
        annotations_dir=args.annotations_dir
    )

    p.preprocess()
