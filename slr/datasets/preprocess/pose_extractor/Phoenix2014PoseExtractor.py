import argparse
import os.path

import pandas as pd

from DatasetBasePoseExtractor import DatasetBasePoseExtractor


class Phoenix2014PoseExtractor(DatasetBasePoseExtractor):
    def __init__(self, model, device='cuda:0', save_dir="./pose_pred"):
        super().__init__(model, device, save_dir)

        self.dataset_name = "Phoenix2014"

    def init_dataset(self, data_dir, features_dir=None, annotations_dir=None):
        self.data_dir = os.path.abspath(data_dir)

        # Ensure all directory paths are set correctly
        self.features_dir = os.path.join(self.data_dir, 'phoenix-2014-multisigner/features/fullFrame-210x260px') \
            if features_dir is None and data_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(self.data_dir, 'phoenix-2014-multisigner/annotations/manual') \
            if annotations_dir is None and data_dir is not None else os.path.abspath(
            annotations_dir) if annotations_dir else None
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        train_corpus = pd.read_csv(os.path.join(self.annotations_dir, 'train.corpus.csv'),
                                   sep='|', header=0, index_col='id')
        dev_corpus = pd.read_csv(os.path.join(self.annotations_dir, 'dev.corpus.csv'),
                                 sep='|', header=0, index_col='id')
        test_corpus = pd.read_csv(os.path.join(self.annotations_dir, 'test.corpus.csv'),
                                  sep='|', header=0, index_col='id')
        self.info = pd.concat(
            [train_corpus.assign(split='train'), dev_corpus.assign(split='dev'), test_corpus.assign(split='test')],
            axis=0, ignore_index=False
        )

    def _get_frames_subdir(self, item):
        return os.path.join(item["split"], item.name, "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        default="td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192",
                        help="Name of the pose estimation model")
    parser.add_argument("--device", type=str,
                        default="cuda:0",
                        help="Device to use for inference")
    parser.add_argument("--save-dir", type=str,
                        default="../../../../data/keypoints/phoenix2014/fullFrame-210x260px",
                        help="Path to the directory where results will be saved")

    parser.add_argument("--dataset-dir", type=str,
                        default="../../../../data/phoenix2014",
                        help="Path to the dataset directory")
    parser.add_argument("--features-dir", type=str,
                        default=None,
                        help="Path to the features directory")
    parser.add_argument("--annotations-dir", type=str,
                        default=None,
                        help="Path to the annotations directory")

    parser.add_argument("--max-workers", type=int,
                        default=4,
                        help="Number of workers for parallel processing")

    parser.add_argument("--json-to-pkl", action='store_true',
                        help="Convert json files to pickle files")

    args = parser.parse_args()

    extractor = Phoenix2014PoseExtractor(args.model, args.device, args.save_dir)
    extractor.init_dataset(args.dataset_dir, args.features_dir, args.annotations_dir)
    extractor.execute(args.max_workers, json_to_pkl=args.json_to_pkl)
