import glob
import os
import subprocess
from abc import abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union

import cv2
import pandas as pd
from tqdm import tqdm
from typing_extensions import LiteralString


class BasePreprocessor:
    def __init__(self, dataset_dir, features_dir=None, annotations_dir=None):
        self.name = "Base"
        self.features_dir = None
        self.annotations_dir = None
        self.info = None

    @abstractmethod
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        pass

    @abstractmethod
    def _check_recognization(self) -> bool:
        pass

    @abstractmethod
    def _check_translation(self) -> bool:
        pass

    @abstractmethod
    def _get_glosses(self, item: pd.Series) -> list[str]:
        pass

    @abstractmethod
    def _get_translation(self, item: pd.Series) -> list[str]:
        pass

    @abstractmethod
    def _get_singer(self, item: pd.Series) -> str:
        pass

    def resize_frames(self, output_dir: str, dsize: tuple = (224, 224), max_workers: int = 4):
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, item in tqdm(self.info.iterrows(), total=len(self.info), desc='Preparing'):
                futures.append(executor.submit(self._resize_one_item_frames, item, output_dir, dsize))
            for future in tqdm(futures, total=len(futures), desc='Resizing frames'):
                future.result()

    def _resize_one_item_frames(self, item, output_dir, dsize: tuple = (224, 224)):
        frames_list = sorted(glob.glob(os.path.join(
            self.features_dir,
            self._get_frames_subdir_filename(item, filename=True)
        )))
        for frame in frames_list:
            img = cv2.imread(frame)
            img_resized = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LANCZOS4)
            saved_file = os.path.join(
                output_dir, f"frames-{dsize[0]}x{dsize[1]}px",
                self._get_frames_subdir_filename(item, filename=False),
                os.path.basename(frame)
            )
            os.makedirs(os.path.dirname(saved_file), exist_ok=True)
            cv2.imwrite(saved_file, img_resized)

    def generate_gloss_vocab(self, output_dir: str):
        self._check_recognization()

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.name.lower()}-gloss-vocab.txt")

        if os.path.exists(output_file):
            overwrite = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("File not overwritten.")
                return False
            else:
                print("Overwriting file...")
                os.remove(output_file)

        sentence_list = [self._get_glosses(item) for _, item in self.info.iterrows()]
        glosses = list({gloss for sentence in sentence_list for gloss in sentence})
        glosses = self._sort_vocab(glosses)

        with open(output_file, "w") as f:
            for gloss in glosses:
                f.write(f"{gloss}\n")

        if os.path.exists(output_file):
            print(f"Sorted gloss vocab file saved at {output_file}")

    def generate_word_vocab(self, output_dir: str):
        self._check_translation()

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.name.lower()}-word-vocab.txt")

        if os.path.exists(output_file):
            overwrite = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("File not overwritten.")
                return False
            else:
                print("Overwriting file...")
                os.remove(output_file)

        sentence_list = [self._get_translation(item) for _, item in self.info.iterrows()]
        words = list({word for sentence in sentence_list for word in sentence})
        words = self._sort_vocab(words)

        with open(output_file, "w") as f:
            for word in words:
                f.write(f"{word}\n")

        if os.path.exists(output_file):
            print(f"Sorted word vocab file saved at {output_file}")

    def _sort_vocab(self, input_list: list) -> list:
        return sorted(input_list)

    def generate_glosses_groundtruth(self, output_dir: str, keep_tmp_file: bool = False):
        self._check_recognization()

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for mode in ['train', 'dev', 'test']:
            info = self.info[self.info['split'] == mode]

            output_file_basename = f"{self.name.lower()}-glosses-groundtruth-{mode}.stm"
            tmp_file_basename = f"tmp.{output_file_basename}"

            output_file = os.path.join(output_dir, output_file_basename)
            tmp_file = os.path.join(output_dir, tmp_file_basename)

            try:
                with open(tmp_file, 'w') as f:
                    for idx, item in info.iterrows():
                        glosses = ' '.join(self._get_glosses(item))
                        line = f"{idx} 1 {self._get_singer(item)} 0.0 1.79769e+308 {glosses}\n"
                        f.write(line)
            except Exception as e:
                print(f"Error writing to file: {e}")
                raise e

            if self._sort_groundtruth_file(tmp_file, output_file):
                print(f"Sorted ground truth file saved at {output_file}")
                if not keep_tmp_file:
                    os.remove(tmp_file)

    def generate_translation_groundtruth(self, output_dir: str, keep_tmp_file: bool = False):
        self._check_translation()

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for mode in ['train', 'dev', 'test']:
            info = self.info[self.info['split'] == mode]

            output_file_basename = f"{self.name.lower()}-translation-groundtruth-{mode}.stm"
            tmp_file_basename = f"tmp.{output_file_basename}"

            output_file = os.path.join(output_dir, output_file_basename)
            tmp_file = os.path.join(output_dir, tmp_file_basename)

            try:
                with open(tmp_file, 'w') as f:
                    for idx, item in info.iterrows():
                        translation = ' '.join(self._get_translation(item))
                        line = f"{idx} 1 {self._get_singer(item)} 0.0 1.79769e+308 {translation}\n"
                        f.write(line)
            except Exception as e:
                print(f"Error writing to file: {e}")
                raise e

            if self._sort_groundtruth_file(tmp_file, output_file):
                print(f"Sorted ground truth file saved at {output_file}")
                if not keep_tmp_file:
                    os.remove(tmp_file)

    def _sort_groundtruth_file(self, input_file, output_file):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found at {input_file}")
        if os.path.exists(output_file):
            overwrite = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("File not overwritten.")
                return False
            else:
                print("Overwriting file...")
                os.remove(output_file)

        try:
            sort_cmd = [
                "sort",
                "-k1,1",
                input_file,
                "-o", output_file
            ]
            subprocess.run(sort_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e.cmd}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise e
        return os.path.exists(output_file)
