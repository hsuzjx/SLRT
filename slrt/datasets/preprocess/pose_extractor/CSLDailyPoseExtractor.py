import argparse
import os.path
import pickle

import pandas as pd

from DatasetBasePoseExtractor import DatasetBasePoseExtractor


class CSLDailyPoseExtractor(DatasetBasePoseExtractor):
    def __init__(self, model, device='cuda:0', save_dir="./pose_pred"):
        super().__init__(model, device, save_dir)

        self.dataset_name = "CSL-Daily"

    def init_dataset(self, data_dir, features_dir=None, annotations_dir=None, split_file=None):
        self.data_dir = os.path.abspath(data_dir)

        # Ensure all directory paths are set correctly
        self.features_dir = os.path.join(self.data_dir, 'sentence_frames-512x512/frames_512x512') \
            if features_dir is None and data_dir is not None else os.path.abspath(
            features_dir) if features_dir else None
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Features directory not found at {self.features_dir}")

        # Set and validate annotation directory
        self.annotations_dir = os.path.join(self.data_dir, 'sentence_label') \
            if annotations_dir is None and data_dir is not None else os.path.abspath(
            annotations_dir) if annotations_dir else None
        if not os.path.exists(self.annotations_dir):
            raise FileNotFoundError(f"Annotations directory not found at {self.annotations_dir}")

        # Set and validate split file
        split_file = os.path.join(self.annotations_dir, "split_1.txt") \
            if split_file is None else os.path.abspath(split_file)
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found at {split_file}")

        # Load and filter samples based on mode
        splits = pd.read_csv(split_file, sep='|', header=0, index_col='name')
        if "S000005_P0004_T00" in splits.index and "S000007_P0003_T00" not in splits.index:
            splits = splits.rename(index={"S000005_P0004_T00": "S000007_P0003_T00"})

        # Load annotations and filter by mode
        annotation_file = os.path.join(self.annotations_dir, "csl2020ct_v2.pkl")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        info = pd.DataFrame(data['info']).set_index('name')

        self.info = pd.concat([info, splits], axis=1)

    def _get_frames_subdir(self, item):
        return item.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        default="td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192",
                        help="Name of the pose estimation model")
    parser.add_argument("--device", type=str,
                        default="cuda:0",
                        help="Device to use for inference")
    parser.add_argument("--save-dir", type=str,
                        default="../../../../data/keypoints/csl-daily/sentence_frames-512x512/frames_512x512",
                        help="Path to the directory where results will be saved")

    parser.add_argument("--dataset-dir", type=str,
                        default="../../../../data/csl-daily",
                        help="Path to the dataset directory")
    parser.add_argument("--features-dir", type=str,
                        default=None,
                        help="Path to the features directory")
    parser.add_argument("--annotations-dir", type=str,
                        default=None,
                        help="Path to the annotations directory")
    parser.add_argument("--split-file", type=str,
                        default=None,
                        help="The split file")

    parser.add_argument("--max-workers", type=int,
                        default=4,
                        help="Number of workers for parallel processing")

    parser.add_argument("--json-to-pkl", action='store_true',
                        help="Convert json files to pickle files")
    parser.add_argument("--only-keypoints", action='store_true',
                        help="Only save the information of keypoints")

    args = parser.parse_args()

    extractor = CSLDailyPoseExtractor(args.model, args.device, args.save_dir)
    extractor.init_dataset(args.dataset_dir, args.features_dir, args.annotations_dir, args.split_file)
    extractor.execute(args.max_workers, json_to_pkl=args.json_to_pkl, only_keypoints=args.only_keypoints)
