import glob
import os

import pandas as pd


def check_phoenix2014_dataset(feature_path, annotations_path, saved_path):
    feature_path = os.path.abspath(feature_path)
    annotations_path = os.path.abspath(annotations_path)
    saved_path = os.path.abspath(saved_path)

    for mode in ['train', 'dev', 'test']:
        # 读取注释文件中的数据
        annotations = pd.read_csv(os.path.join(annotations_path, f'{mode}.corpus.csv'), sep='|', header=0,
                                  index_col='id')
        for folder in annotations['folder']:
            source_files = glob.glob(os.path.join(feature_path, mode, folder))
            source_files_basename = [os.path.basename(file) for file in source_files]
            resized_files = glob.glob(os.path.join(saved_path, mode, folder))

            for file in resized_files:
                if os.path.basename(file) not in source_files_basename:
                    print(file)
                # else:
                #     print('ok')


if __name__ == '__main__':
    check_phoenix2014_dataset(
        '../../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-210x260px',
        '../../data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
        '../../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px'
    )
