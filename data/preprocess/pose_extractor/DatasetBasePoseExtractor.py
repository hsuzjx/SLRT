import os.path
import sys
from abc import abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from copyreg import pickle

import mmcv
import mmdet
import mmengine
import mmpose
import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

from PoseExtractor import PoseExtractor


class DatasetBasePoseExtractor(PoseExtractor):
    """
    Base class for extracting pose from dataset.
    """

    def __init__(self, model, device='cuda:0', save_dir="./pose_pred"):
        super().__init__(model, device)

        self.dataset_name = "Base"

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.data_dir = None
        self.features_dir = None
        self.annotations_dir = None
        self.info: pd.DataFrame = None

        self.keypoints = dict()

    @abstractmethod
    def init_dataset(self, data_dir, features_dir=None, annotations_dir=None):
        """
        Initialize dataset.
        """
        pass

    @abstractmethod
    def _get_frames_subdir(self, item):
        pass

    def execute(self, max_workers=4):
        if self.info is None:
            raise ValueError("Dataset not initialized. Please call init_dataset() before executing.")

        self._save_information()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, item in tqdm(self.info.iterrows(), total=len(self.info), desc='Preparing'):
                futures.append(executor.submit(self._extract_pose, item))

            for future in tqdm(futures, total=len(futures), desc='Executing'):
                future.result()

        self._save_pkl()

    def _extract_pose(self, item):
        frames_dir = os.path.join(self.features_dir, self._get_frames_subdir(item))
        pred_out_dir = os.path.join(self.save_dir, self._get_frames_subdir(item))
        results = self(frames_dir, show=False, pred_out_dir=pred_out_dir)

        self.keypoints[item.name] = results

    def _save_information(self):
        information_file = os.path.join(self.save_dir, 'information.txt')
        with open(information_file, "w") as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Features directory: {self.features_dir}\n")
            f.write(f"Annotations directory: {self.annotations_dir}\n\n")
            f.write(f"Pose Estimation Model: {self.model}\n\n")
            f.write(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
            f.write(f"PyTorch version: {torch.__version__}\n")
            f.write(f"Torchvision version: {torchvision.__version__}\n")
            f.write(f"mmpose version: {mmpose.__version__}\n")
            f.write(f"mmengine version: {mmengine.__version__}\n")
            f.write(f"mmdet version: {mmdet.__version__}\n")
            f.write(f"mmcv version: {mmcv.__version__}\n")

    def _save_pkl(self):
        assert len(self.keypoints) == len(self.info)

        save_data = {}
        for name in self.keypoints.keys():
            video_results = self.keypoints[name]

            video_kps_list = []
            for frame_result in video_results:
                if len(frame_result["predictions"][0]) > 1:
                    print("Warning: more than one person detected in the frame")

                keypoints = frame_result["predictions"][0][0]['keypoints']
                keypoints_scores = frame_result["predictions"][0][0]['keypoints_scores']

                video_kps_list.append([keypoints[i] + [keypoints_scores[i]] for i in range(len(keypoints))])

            kps_array = np.array(video_kps_list, dtype=np.float32)
            save_data[name] = {'keypoints': kps_array}

        assert len(save_data) == len(self.info)

        pkl_file = os.path.join(self.save_dir, f'{self.dataset_name.lower()}_keypoints.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(save_data, f)
