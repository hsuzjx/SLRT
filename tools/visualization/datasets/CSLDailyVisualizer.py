import glob
import os

import pandas as pd
from typing_extensions import override

from tools.visualization.datasets.DatasetBaseVisualizer import DatasetBaseVisualizer
from tools.visualization.datasets.utils import safe_pickle_load


class CSLDailyVisualizer(DatasetBaseVisualizer):
    def __init__(self, data_dir, keypoints_file, temp_dir):
        super().__init__(data_dir, keypoints_file, temp_dir)

    @override
    def interface_fn(self, idx: str):
        item = self._get_item(idx)
        video_file = self._frames_to_video(item)

        return video_file, " ".join(item["label_gloss"]), "".join(item["label_word"]), item

    @override
    def _init_data_params(self):
        self.frames_dir = os.path.join(self.data_dir, 'sentence_frames-512x512', 'frames_512x512')
        data_manager = safe_pickle_load(os.path.join(self.data_dir, 'sentence_label', 'csl2020ct_v2.pkl'))
        self.info = pd.DataFrame(data_manager['info'])
        self.info.set_index("name", inplace=True)
        self.dataset_size = len(self.info)

    @override
    def _get_frames(self, item: pd.DataFrame) -> list:
        frame_file_list = sorted(glob.glob(os.path.join(self.frames_dir, item.name, '*.jpg')))
        if not frame_file_list:
            raise FileNotFoundError(f"No frames found in directory: {self.frames_dir}")
        return frame_file_list

    @override
    def _get_edges(self, name: str) -> list:
        pass
