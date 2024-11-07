import argparse
import atexit
import glob
import os

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from tools.visualization.datasets.utils import TempDirManager, safe_pickle_load


class CSLDailyVisualizer:
    """
    Visualizer for CSL Daily dataset.

    This class provides a Gradio interface for visualizing video samples from the CSL Daily dataset.
    
    Attributes
    ----------
    data_dir (str): Path to the data directory.
    frames_dir (str): Path to the directory containing the frames.
    data_manager: The dataset manager object.
    info (pd.DataFrame): DataFrame containing the dataset information.
    dataset_size (int): Size of the dataset.
    keypoints_data: The keypoints data object.
    temp_dir_manager: The temporary directory manager object.
    temp_dir (str): Path to the temporary directory.
    confidence_threshold (float): Confidence threshold for keypoints.
    sigma (int): Sigma for Gaussian kernel.
    fps (int): Frames per second.

    Methods
    -------
    __init__(self, data_dir, temp_dir, confidence_threshold=0.3, sigma=10, fps=25):
        Initializes the CSLDailyVisualizer with the given parameters.

    _get_item(self, idx: str):
        Retrieves an item from the dataset by index or name.

    _get_frame_file_list(self, name: str):
        Retrieves the list of frame files for a given sample name.

    _frames_to_video(self, name: str, frame_file_list: list, add_keypoints=False, add_heatmap=False):
        Converts a list of frame files into a video.

    _add_keypoints_to_frame(self, frame, name, frame_idx):
        Adds keypoints to a frame.

    _add_heatmap_to_frame(self, frame, name, frame_idx):
        Adds a heatmap to a frame.
    """

    def __init__(self, data_dir, temp_dir, confidence_threshold=0.3, sigma=10, fps=25):
        """
        Initializes the CSLDailyVisualizer with the given parameters.

        Loads the dataset and initializes the temporary directory manager.

        Args:
            data_dir (str): Path to the data directory.
            temp_dir (str): Path to the temporary directory or None to auto-create.
            confidence_threshold (float, optional): Confidence threshold for keypoints. Defaults to 0.3.
            sigma (int, optional): Sigma for Gaussian kernel. Defaults to 10.
            fps (int, optional): Frames per second. Defaults to 25.
        """
        self.data_dir = os.path.abspath(data_dir)
        self.frames_dir = os.path.join(self.data_dir, 'sentence_frames-512x512', 'frames_512x512')
        self.data_manager = safe_pickle_load(os.path.join(self.data_dir, 'sentence_label', 'csl2020ct_v2.pkl'))
        self.info = pd.DataFrame(self.data_manager['info'])
        self.info.set_index("name", inplace=True)
        self.dataset_size = len(self.data_manager['info'])
        self.keypoints_data = safe_pickle_load(os.path.join(self.data_dir, 'csl-daily-keypoints.pkl'))

        self.temp_dir_manager = TempDirManager(temp_dir)
        self.temp_dir = self.temp_dir_manager.get_temp_dir()

        self.confidence_threshold = confidence_threshold
        self.sigma = sigma
        self.fps = fps

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

    def _get_frame_file_list(self, name: str):
        """
        Retrieves the list of frame files for a given sample name.

        Args:
            name (str): Name of the sample.

        Returns:
            list: List of frame file paths.

        Raises:
            FileNotFoundError: If no frames are found in the directory.
        """
        frame_file_list = sorted(glob.glob(os.path.join(self.frames_dir, name, '*.jpg')))
        if not frame_file_list:
            raise FileNotFoundError(f"No frames found in directory: {self.frames_dir}")
        return frame_file_list

    def _frames_to_video(self, name: str, frame_file_list: list, add_keypoints=False, add_heatmap=False):
        """
        Converts a list of frame files into a video.

        Args:
            name (str): Name of the sample.
            frame_file_list (list): List of frame file paths.
            add_keypoints (bool, optional): Whether to add keypoints to the frames. Defaults to False.
            add_heatmap (bool, optional): Whether to add heatmaps to the frames. Defaults to False.

        Returns:
            str: Path to the output video file.

        Raises:
            FileNotFoundError: If any frame file is not found.
        """
        output_file = os.path.join(self.temp_dir, f"{name}.mp4")

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

            if add_keypoints:
                self._add_keypoints_to_frame(frame, name, frame_idx)

            if add_heatmap:
                frame = self._add_heatmap_to_frame(frame, name, frame_idx)

            video_writer.write(frame)

        video_writer.release()

        return output_file

    def _add_keypoints_to_frame(self, frame, name, frame_idx):
        """
        Adds keypoints to a frame.

        Args:
            frame (np.ndarray): Frame to add keypoints to.
            name (str): Name of the sample.
            frame_idx (int): Index of the frame.
        """
        for kp in self.keypoints_data[name]['keypoints'][frame_idx]:
            x, y, confidence = kp
            if confidence > self.confidence_threshold and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    def _add_heatmap_to_frame(self, frame, name, frame_idx):
        """
        Adds a heatmap to a frame.

        Args:
            frame (np.ndarray): Frame to add heatmap to.
            name (str): Name of the sample.
            frame_idx (int): Index of the frame.

        Returns:
            np.ndarray: Frame with heatmap added.
        """
        heatmap = self.create_heatmap(
            image=frame,
            keypoints=self.keypoints_data[name]['keypoints'][frame_idx],
            confidence_threshold=self.confidence_threshold,
            sigma=self.sigma
        )
        alpha = 0.5
        frame = cv2.addWeighted(frame, alpha, heatmap, 1 - alpha, 0)
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

    def gr_fn(self, idx: str, add_keypoints: bool, add_heatmap: bool):
        """
        Gradio function for visualizing a sample.

        Args:
            idx (str): Index or name of the sample.
            add_keypoints (bool): Whether to add keypoints to the frames.
            add_heatmap (bool): Whether to add heatmaps to the frames.

        Returns:
            tuple: Path to the output video file, translation, and additional information.
        """
        item = self._get_item(idx)
        frame_file_list = self._get_frame_file_list(item.name)
        video_file = self._frames_to_video(item.name, frame_file_list, add_keypoints, add_heatmap)

        translation = "".join(item["label_word"])

        return video_file, translation, item

    def launch(self, server_name, server_port, share=False, inbrowser=False):
        """
        Launches the Gradio interface.

        Args:
            server_name (str): Server name.
            server_port (int): Server port.
            share (bool, optional): Whether to share the visualization. Defaults to False.
            inbrowser (bool, optional): Whether to open the browser on launch. Defaults to False.
        """
        # Initialize Gradio interface
        iface = gr.Interface(
            title='CSL Daily Visualizer',
            fn=self.gr_fn,
            inputs=[
                gr.Text(label='Sample', info=f'sample name or index({0}~{self.dataset_size - 1})'),
                gr.Checkbox(label='Add Keypoints?', value=False),
                gr.Checkbox(label='Add Heatmap?', value=False)
            ],
            outputs=[
                gr.Video(label='Video', autoplay=True, show_download_button=True),
                gr.Text(label='Translation'),
                gr.Text(label='Info')
            ],
            live=False,
            flagging_mode='never',
        )

        # Launch Gradio interface service
        iface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            inbrowser=inbrowser
        )


if __name__ == "__main__":
    # Create an argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="CSL Daily Visualizer")

    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Path to the temporary directory or None to auto-create")

    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Confidence threshold for keypoints")
    parser.add_argument("--sigma", type=int, default=10, help="Sigma for Gaussian kernel")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")

    parser.add_argument("--host", type=str, default="localhost", help="Server name")
    parser.add_argument("--port", type=int, default=28080, help="Server port")
    parser.add_argument("--browse", action="store_true", help="Open browser on launch")
    parser.add_argument("--share", action="store_false", help="Share the visualization")

    # Parse command-line arguments
    args = parser.parse_args()

    # Initialize CSL Daily Visualizer instance
    visualizer = CSLDailyVisualizer(
        data_dir=args.data_dir,
        temp_dir=args.temp_dir,
        confidence_threshold=args.confidence_threshold,
        sigma=args.sigma,
        fps=args.fps,
    )
    # Launch the Gradio interface
    visualizer.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.browse
    )

    # Register a function to clean up the temporary directory on program exit
    atexit.register(visualizer.temp_dir_manager.clean_tmp_dir)

    # Start a repeating task to clean up the temporary directory every 600 seconds
    visualizer.temp_dir_manager.start_repeat_delete(600)

    # Get the size of the dataset
    data_len = len(visualizer.data_manager['info'])
