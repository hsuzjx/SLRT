import glob
import os

import pandas as pd
from typing_extensions import override

from tools.visualization.datasets.DatasetBaseVisualizer import DatasetBaseVisualizer


class Phoenix2014Visualizer(DatasetBaseVisualizer):
    def __init__(self, data_dir, keypoints_file, temp_dir):
        super().__init__(data_dir, keypoints_file, temp_dir)

    @override
    def interface_fn(self, idx: str):
        item = self._get_item(idx)
        video_file = self._frames_to_video(item)

        return video_file, " ".join([gloss for gloss in item['annotation'].split(' ') if gloss]), "None", item

    @override
    def _init_data_params(self):
        self.frames_dir = os.path.join(self.data_dir, 'phoenix-2014-multisigner/features/fullFrame-210x260px')
        annotations_dir = os.path.join(self.data_dir, 'phoenix-2014-multisigner/annotations/manual')
        train_corpus = pd.read_csv(os.path.join(annotations_dir, 'train.corpus.csv'), sep='|', header=0, index_col='id')
        dev_corpus = pd.read_csv(os.path.join(annotations_dir, 'dev.corpus.csv'), sep='|', header=0, index_col='id')
        test_corpus = pd.read_csv(os.path.join(annotations_dir, 'test.corpus.csv'), sep='|', header=0, index_col='id')
        self.info = pd.concat(
            [train_corpus.assign(split='train'), dev_corpus.assign(split='dev'), test_corpus.assign(split='test')],
            axis=0, ignore_index=False
        )
        self.dataset_size = len(self.info)

    @override
    def _get_frames(self, item: pd.DataFrame) -> list:
        frame_file_list = sorted(glob.glob(os.path.join(self.frames_dir, item["split"], item.name, "1", '*.png')))
        if not frame_file_list:
            raise FileNotFoundError(f"No frames found in directory: {self.frames_dir}")
        return frame_file_list

    @override
    def _get_edges(self, name: str) -> list:
        pass
