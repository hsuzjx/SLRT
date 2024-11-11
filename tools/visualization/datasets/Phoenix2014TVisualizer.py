import os

import pandas as pd
from typing_extensions import override

from tools.visualization.datasets.DatasetBaseVisualizer import DatasetBaseVisualizer


class Phoenix2014TVisualizer(DatasetBaseVisualizer):
    def __init__(self, dataset_dir, keypoints_file, temp_dir):
        super().__init__(dataset_dir, keypoints_file, temp_dir)

    @override
    def interface_fn(self, idx: str):
        item = self._get_item(idx)
        video_file = self._frames_to_video(item)

        return (
            video_file,
            " ".join([gloss for gloss in item['orth'].split(' ') if gloss]),
            " ".join([word for word in item['translation'].split(' ') if word]),
            item
        )

    @override
    def _init_data_params(self):
        self.features_dir = os.path.join(self.data_dir, 'PHOENIX-2014-T/features/fullFrame-210x260px')
        annotations_dir = os.path.join(self.data_dir, 'PHOENIX-2014-T/annotations/manual')

        train_corpus = pd.read_csv(
            os.path.join(annotations_dir, 'PHOENIX-2014-T.train.corpus.csv'),
            sep='|', header=0, index_col='name'
        )
        dev_corpus = pd.read_csv(
            os.path.join(annotations_dir, 'PHOENIX-2014-T.dev.corpus.csv'),
            sep='|', header=0, index_col='name'
        )
        test_corpus = pd.read_csv(
            os.path.join(annotations_dir, 'PHOENIX-2014-T.test.corpus.csv'),
            sep='|', header=0, index_col='name'
        )

        self.info = pd.concat(
            [train_corpus.assign(split='train'), dev_corpus.assign(split='dev'), test_corpus.assign(split='test')],
            axis=0, ignore_index=False
        )

        self.dataset_size = len(self.info)

    @override
    def _get_frames_subdir(self, item: pd.DataFrame):
        return os.path.join(item["split"], item.name, '*.png')
