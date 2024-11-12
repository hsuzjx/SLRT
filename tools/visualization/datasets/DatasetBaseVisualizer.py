import glob
import os
from abc import abstractmethod

import cv2
import numpy as np
import pandas as pd

from tools.visualization.datasets.utils import safe_pickle_load


class DatasetBaseVisualizer:
    """
    Base class for dataset visualizers.

    This class provides a base structure for visualizing datasets using Gradio.

    Attributes
    ----------
    data_dir (str): Path to the data directory.
    features_dir (str): Path to the directory of features.
    info (pd.DataFrame): DataFrame containing the dataset information.
    dataset_size (int): Size of the dataset.
    keypoints (dict): Dictionary containing the keypoints data.
    temp_dir (str): Path to the temporary directory.
    fps (int): Frames per second.
    confidence_threshold (float): Confidence threshold for keypoints.
    keypoints_color (tuple): Color of the keypoints.
    keypoints_scale (float): Scale of the keypoints.
    keypoints_style (str): Style of the keypoints.
    keypoints_thickness (int): Thickness of the keypoints.
    edges (dict): Dict of edges for connecting keypoints.
    edges_color (tuple): Color of the edges.
    edges_thickness (int): Thickness of the edges.
    heatmap_sigma (int): Sigma for Gaussian kernel used in heatmap generation.
    heatmap_alpha (float): Alpha blending factor for heatmap overlay.
    heatmap_gamma (float): Gamma correction factor for heatmap overlay.
    add_keypoints (bool): Flag to indicate whether to add keypoints to the frames.
    add_edges (bool): Flag to indicate whether to add edges to the frames.
    add_heatmap (bool): Flag to indicate whether to add heatmaps to the frames.

    Methods
    -------
    __init__(self, data_dir, keypoints_file, temp_dir):
        Initializes the DatasetBaseVisualizer with the given parameters.

    init_fn(self, **kwargs):
        Initializes visualization parameters.

    interface_fn(self, idx: str):
        Abstract method to define the interface function.

    _init_data_params(self):
        Abstract method to initialize data parameters.

    _init_edges(self):
        Abstract method to initialize edges.

    _get_frames_subdir(self, item: pd.DataFrame) -> str:
        Abstract method to get the frames subdir for a given sample.

    _get_item(self, idx: str):
        Retrieves an item from the dataset by index or name.

    _frames_to_video(self, item: pd.DataFrame):
        Converts a list of frame files into a video.

    _add_keypoints_to_frame(self, frame, frame_keypoints):
        Adds keypoints to a frame.

    _add_edges_to_frame(self, frame, frame_keypoints):
        Adds edges to a frame.

    _add_heatmap_to_frame(self, frame, frame_keypoints):
        Adds a heatmap to a frame.

    @staticmethod
    create_heatmap(image, keypoints, confidence_threshold=0.3, sigma=10):
        Creates a heatmap from keypoints.
    """

    def __init__(self, data_dir, keypoints_file, temp_dir):
        """
        Initializes the DatasetBaseVisualizer with the given parameters.

        Args:
            data_dir (str): Path to the data directory.
            keypoints_file (str): Path to the keypoints file.
            temp_dir (str): Path to the temporary directory or None to auto-create.
        """
        # Data params
        self.data_dir = os.path.abspath(data_dir)
        self.features_dir = None
        self.info = None
        self.dataset_size = None
        self._init_data_params()

        # Temp params
        self.temp_dir = temp_dir

        # Visualization params
        self.fps = 25
        self.confidence_threshold = 0.3

        # Keypoints params
        self.keypoints = safe_pickle_load(keypoints_file)
        self.keypoints_color = (0, 0, 255)
        self.keypoints_scale = 1
        self.keypoints_style = '*'
        self.keypoints_thickness = 2

        # Edges params
        self.edges = dict()
        self._init_edges()
        self.edges_color = (255, 0, 0)
        self.edges_thickness = 1

        # Heatmap params
        self.heatmap_sigma = 10
        self.heatmap_alpha = 0.5
        self.heatmap_gamma = 0

        # Visualization flags
        self.add_keypoints = False
        self.add_edges = False
        self.add_heatmap = False

    def init_fn(
            self,
            fps=25,
            confidence_threshold=0.3,
            keypoints_color=(0, 0, 255),
            keypoints_scale=1,
            keypoints_style='*',
            keypoints_thickness=2,
            edges_color=(255, 0, 0),
            edges_thickness=1,
            heatmap_sigma=10,
            heatmap_alpha=0.5,
            heatmap_gamma=0,
            add_keypoints=False,
            add_edges=False,
            add_heatmap=False
    ):
        """
        Initializes visualization parameters.

        Args:
            fps (int, optional): Frames per second. Defaults to 25.
            confidence_threshold (float, optional): Confidence threshold for keypoints. Defaults to 0.3.
            keypoints_color (tuple, optional): Color of the keypoints. Defaults to (0, 0, 255).
            keypoints_scale (float, optional): Scale of the keypoints. Defaults to 1.
            keypoints_style (str, optional): Style of the keypoints. Defaults to '*'.
            keypoints_thickness (int, optional): Thickness of the keypoints. Defaults to 2.
            edges_color (tuple, optional): Color of the edges. Defaults to (255, 0, 0).
            edges_thickness (int, optional): Thickness of the edges. Defaults to 1.
            heatmap_sigma (int, optional): Sigma for Gaussian kernel used in heatmap generation. Defaults to 10.
            heatmap_alpha (float, optional): Alpha blending factor for heatmap overlay. Defaults to 0.5.
            heatmap_gamma (float, optional): Gamma correction factor for heatmap overlay. Defaults to 0.
            add_keypoints (bool, optional): Flag to indicate whether to add keypoints to the frames. Defaults to False.
            add_edges (bool, optional): Flag to indicate whether to add edges to the frames. Defaults to False.
            add_heatmap (bool, optional): Flag to indicate whether to add heatmaps to the frames. Defaults to False.
        """
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.keypoints_color = keypoints_color
        self.keypoints_scale = keypoints_scale
        self.keypoints_style = keypoints_style
        self.keypoints_thickness = keypoints_thickness
        self.edges_color = edges_color
        self.edges_thickness = edges_thickness
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_gamma = heatmap_gamma
        self.add_keypoints = add_keypoints
        self.add_edges = add_edges
        self.add_heatmap = add_heatmap

    @abstractmethod
    def interface_fn(self, idx: str):
        """
        Abstract method to define the interface function.

        Args:
            idx (str): Index or name of the item to visualize.
        """
        pass

    @abstractmethod
    def _init_data_params(self):
        """
        Abstract method to initialize data parameters.
        """
        pass

    @abstractmethod
    def _get_frames_subdir(self, item: pd.DataFrame):
        """
        Abstract method to get the list of frame files for a given sample.

        Args:
            item (pd.DataFrame): The item to get frames for.

        Returns:
            frames_subdir (str): The frames subdir for the given sample.
        """
        pass

    def _init_edges(self):
        """
        Abstract method to initialize edges.
        """
        body_kpts = list(range(1, 18))
        left_foot_kpts = list(range(18, 21))
        right_foot_kpts = list(range(21, 24))
        face_kpts = list(range(24, 92))
        left_hand_kpts = list(range(92, 113))
        raght_hand_kpts = list(range(113, 134))

        self.edges = {
            "body_skeleton": [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ],
            "left_hand_skeleton": [
                [92, 93], [92, 97], [92, 101], [92, 105], [92, 109],  # root
                [93, 94], [94, 95], [95, 96],  # finger 1
                [97, 98], [98, 99], [99, 100],  # finger 2
                [101, 102], [102, 103], [103, 104],  # finger 3
                [105, 106], [106, 107], [107, 108],  # finger 4
                [109, 110], [110, 111], [111, 112]  # finger 5
            ],
            "right_hand_skeleton": [
                [113, 114], [113, 118], [113, 122], [113, 126], [113, 130],  # root
                [114, 115], [115, 116], [116, 117],  # finger 1
                [118, 119], [119, 120], [120, 121],  # finger 2
                [122, 123], [123, 124], [124, 125],  # finger 3
                [126, 127], [127, 128], [128, 129],  # finger 4
                [130, 131], [131, 132], [132, 133]  # finger 5
            ]
        }

    def _get_item(self, idx: str):
        """
        Retrieves an item from the dataset by index or name.

        Args:
            idx (str): Index or name of the item to retrieve.

        Returns:
            pd.Series: The retrieved item.

        Raises:
            ValueError: If the index is out of range.
            KeyError: If the name is not found.
        """
        if idx.isdigit():
            idx = int(idx)
            if idx < 0 or idx >= self.dataset_size:
                raise ValueError("Invalid index")
            return self.info.iloc[idx]
        else:
            return self.info.loc[idx]

    def _frames_to_video(self, item: pd.DataFrame):
        """
        Converts a list of frame files into a video.

        Args:
            item (pd.DataFrame): The item to convert to a video.

        Returns:
            str: Path to the output video file.

        Raises:
            ValueError: If the frame file list is empty.
            FileNotFoundError: If any frame file is not found.
        """
        frames_subdir = self._get_frames_subdir(item)
        frame_file_list = sorted(glob.glob(os.path.join(self.features_dir, frames_subdir)))
        if not frame_file_list:
            raise ValueError("Frame file list is empty.")
        output_file = os.path.join(self.temp_dir, f"{item.name}.mp4")

        first_frame = cv2.imread(frame_file_list[0])
        if first_frame is None:
            raise FileNotFoundError(f"Frame file {frame_file_list[0]} not found.")
        frame_size = (first_frame.shape[1], first_frame.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_file, fourcc, self.fps, frame_size)

        for frame_idx, frame_file in enumerate(frame_file_list):
            frame = cv2.imread(frame_file)
            if frame is None:
                raise FileNotFoundError(f"Frame file {frame_file} not found.")

            if self.keypoints:
                frame_keypoints = self.keypoints[item.name]['keypoints'][frame_idx]
                if self.add_keypoints:
                    frame = self._add_keypoints_to_frame(frame, frame_keypoints)
                if self.add_edges:
                    frame = self._add_edges_to_frame(frame, frame_keypoints)
                if self.add_heatmap:
                    frame = self._add_heatmap_to_frame(frame, frame_keypoints)

            video_writer.write(frame)

        video_writer.release()

        return output_file

    def _add_keypoints_to_frame(self, frame, frame_keypoints):
        """
        Adds keypoints to a frame.

        Args:
            frame (np.ndarray): Frame to add keypoints to.
            frame_keypoints (list): List of keypoints for the frame.

        Returns:
            np.ndarray: Frame with keypoints added.
        """
        for kp_idx, kp in enumerate(frame_keypoints):
            x, y, confidence = kp

            if confidence > self.confidence_threshold and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                if self.keypoints_style == 'Number':
                    text = str(kp_idx + 1)
                else:
                    text = self.keypoints_style

                (text_width, text_height), _ = cv2.getTextSize(
                    text=text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=self.keypoints_scale,
                    thickness=2
                )
                text_x = int(x - text_width / 2)
                text_y = int(y + text_height / 2)

                cv2.putText(
                    frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=self.keypoints_scale,
                    color=self.keypoints_color,
                    thickness=self.keypoints_thickness,
                    bottomLeftOrigin=False
                )

        return frame

    def _add_edges_to_frame(self, frame, frame_keypoints):
        """
        Adds edges to a frame.

        Args:
            frame (np.ndarray): Frame to add edges to.
            frame_keypoints (list): List of keypoints for the frame.

        Returns:
            np.ndarray: Frame with edges added.
        """
        for key in self.edges.keys():
            edges = self.edges[key]

            for edge in edges:
                x1, y1, c1 = frame_keypoints[edge[0] - 1]
                x2, y2, c2 = frame_keypoints[edge[1] - 1]
                if c1 > self.confidence_threshold and c2 > self.confidence_threshold:
                    cv2.line(
                        frame,
                        pt1=(int(x1), int(y1)),
                        pt2=(int(x2), int(y2)),
                        color=self.edges_color,
                        thickness=self.edges_thickness
                    )

        return frame

    def _add_heatmap_to_frame(self, frame, frame_keypoints):
        """
        Adds a heatmap to a frame.

        Args:
            frame (np.ndarray): Frame to add heatmap to.
            frame_keypoints (list): List of keypoints for the frame.

        Returns:
            np.ndarray: Frame with heatmap added.
        """
        heatmap = self.create_heatmap(
            image=frame,
            keypoints=frame_keypoints,
            confidence_threshold=self.confidence_threshold,
            sigma=self.heatmap_sigma
        )
        frame = cv2.addWeighted(frame, self.heatmap_alpha, heatmap, 1 - self.heatmap_alpha, self.heatmap_gamma)
        return frame

    @staticmethod
    def create_heatmap(image, keypoints, confidence_threshold=0.3, sigma=10):
        """
        Creates a heatmap from keypoints.

        Args:
            image (np.ndarray): Image to create heatmap for.
            keypoints (list): List of keypoints.
            confidence_threshold (float, optional): Confidence threshold for keypoints. Defaults to 0.3.
            sigma (int, optional): Sigma for Gaussian kernel. Defaults to 10.

        Returns:
            np.ndarray: Heatmap.
        """
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Prepare Gaussian kernel
        g = cv2.getGaussianKernel(int(sigma * 6), sigma)
        g = g * g.T

        for kp in keypoints:
            x, y, confidence = kp
            if confidence > confidence_threshold and 0 <= x < width and 0 <= y < height:
                x, y = int(x), int(y)
                x_start = np.clip(x - 3 * sigma, 0, width - 1)
                y_start = np.clip(y - 3 * sigma, 0, height - 1)
                x_end = np.clip(x + 3 * sigma, 0, width - 1)
                y_end = np.clip(y + 3 * sigma, 0, height - 1)

                heatmap[y_start:y_end, x_start:x_end] = np.maximum(
                    heatmap[y_start:y_end, x_start:x_end],
                    g[:y_end - y_start, :x_end - x_start] * confidence)

        # Normalize heatmap and convert to color map
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap
