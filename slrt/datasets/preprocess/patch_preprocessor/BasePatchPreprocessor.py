import glob
import os
from abc import abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import LiteralString


class BasePatchPreprocessor:
    def __init__(self):
        self.name = "Base"
        self.features_dir = None
        self.annotations_dir = None
        self.kps_file = None
        self.info = None
        self.kps_info = None

    @abstractmethod
    def _get_frames_subdir_filename(
            self,
            item: pd.DataFrame,
            filename: bool = True
    ) -> Union[LiteralString, str, bytes]:
        pass

    def ge_patches(self, output_dir: str, patch_hw: tuple = (13, 13), max_workers: int = 4):
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        half_patch_h = (patch_hw[0] - 1) // 2
        half_patch_w = (patch_hw[1] - 1) // 2

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, item in tqdm(self.info.iterrows(), total=len(self.info), desc='Preparing'):
                futures.append(
                    executor.submit(
                        self._ge_one_item_patches,
                        item, self.kps_info[item.name]["keypoints"], output_dir, half_patch_h, half_patch_w
                    )
                )
            for future in tqdm(futures, total=len(futures), desc='Processing'):
                future.result()

    def _ge_one_item_patches(
            self,
            item: pd.DataFrame,
            kps,
            output_dir: str,
            half_patch_h,
            half_patch_w
    ):
        frames_list = sorted(glob.glob(os.path.join(
            self.features_dir,
            self._get_frames_subdir_filename(item, filename=True)
        )))

        for t in range(kps.shape[0]):
            frame = frames_list[t]
            frame_name = os.path.basename(frame).split('.')
            frame_name = '.'.join(frame_name[0:-1])

            img = cv2.imread(frame)
            x_max = img.shape[1]
            y_max = img.shape[0]

            img = np.pad(
                img,
                ((half_patch_h, half_patch_h), (half_patch_w, half_patch_w), (0, 0)),
                mode='constant',
                constant_values=0
            )

            for v in range(kps.shape[1]):
                kp = kps[t, v]
                x, y, c = kp
                x = int(x)
                y = int(y)
                if 0 <= x <= x_max and 0 <= y <= y_max:
                    patch = img[y:y + 2 * half_patch_h + 1, x:x + 2 * half_patch_w + 1, :]
                else:
                    patch = np.zeros((2 * half_patch_h + 1, 2 * half_patch_w + 1, 3), dtype=np.uint8)

                saved_file = os.path.join(
                    output_dir, f"patches-{half_patch_h * 2 + 1}x{half_patch_w * 2 + 1}px",
                    self._get_frames_subdir_filename(item, filename=False),
                    frame_name, f"{v:03d}.png"
                )
                os.makedirs(os.path.dirname(saved_file), exist_ok=True)

                cv2.imwrite(saved_file, patch)
                # with open(os.path.join(os.path.dirname(saved_file), f"{v:03d}.txt"), "w") as f:
                #     f.write(f"{x} {y} {c}, {aaa}\n {patch}")
