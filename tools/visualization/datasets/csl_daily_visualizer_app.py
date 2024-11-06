import argparse
import atexit
import glob
import os
import tempfile

import cv2
import gradio as gr
import numpy as np

from tools.visualization.datasets.utils import VideoWriter, TempDirManager, safe_pickle_load


class CSLDailyVisualizer:
    """
    用于处理和可视化连续手语识别(CSL)数据的类。
    负责从帧生成视频，可选择性地添加关键点和热图。
    """

    def __init__(self, data_dir, temp_dir, server_name, server_port, confidence_threshold=0.3, sigma=10, fps=25):
        """
        初始化CSLDailyVisualizer类的实例。

        参数:
        - data_dir: 存储数据的目录路径。
        - temp_dir: 用于临时文件的目录路径。
        - server_name: 服务器的名称。
        - server_port: 服务器的端口号。
        - confidence_threshold: 关键点置信度阈值，默认为0.3。
        - sigma: 高斯模糊的sigma参数，默认为10。
        """
        # 保存传入的参数
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        self.server_name = server_name
        self.server_port = server_port
        self.confidence_threshold = confidence_threshold
        self.sigma = sigma
        self.fps = fps

        self.frames_dir = os.path.join(self.data_dir, 'sentence_frames-512x512', 'frames_512x512')

        # 加载数据管理器和初始化临时目录管理器与视频写入器
        # 从数据目录加载预处理的数据
        self.data_manager = safe_pickle_load(os.path.join(data_dir, 'sentence_label', 'csl2020ct_v2.pkl'))
        self.keypoints_data = safe_pickle_load(os.path.join(data_dir, 'csl-daily-keypoints.pkl'))
        # 初始化临时目录管理器，用于存储临时文件
        self.temp_dir_manager = TempDirManager(temp_dir)

    def process_frames_to_video(self, idx, is_add_keypoints, is_add_heatmap):
        """
        将帧处理成视频，根据索引从数据中加载帧，并根据标志添加关键点或热图。

        参数:
        - idx: 要处理的数据索引。
        - is_add_keypoints: 布尔值，指示是否在视频中添加关键点。
        - is_add_heatmap: 布尔值，指示是否在视频中添加热图。

        返回:
        - output_video_path: 输出视频的文件路径。
        - label: 与视频关联的标签词。
        - info: 当前数据的信息。
        """

        # 确保数据有效且索引在范围内
        if 'info' not in self.data_manager or idx < 0 or idx >= len(self.data_manager['info']):
            raise ValueError("Invalid data or index")

        # 根据索引获取名称和帧路径
        name = self.data_manager['info'][idx]['name']
        frames = sorted(glob.glob(os.path.join(self.frames_dir, name, '*.jpg')))
        if not frames:
            raise FileNotFoundError(f"No frames found for {name}")

        # 读取第一帧以获取帧尺寸
        first_frame = cv2.imread(frames[0])
        if first_frame is None:
            raise IOError(f"Failed to read the first frame from {frames[0]}")

        # 初始化视频写入器
        video_writer = VideoWriter(output_file=os.path.join(self.temp_dir, f"{name}.mp4"), fps=self.fps)
        video_writer.init_writer(first_frame)

        # 遍历所有帧，进行处理并添加到视频中
        for i, f in enumerate(frames):
            frame = cv2.imread(f)
            if frame is None:
                print(f"Error reading frame: {f}")
                continue

            # 根据标志添加关键点和/或热图
            if is_add_keypoints:
                self._add_keypoints_to_frame(frame, name, i)
            if is_add_heatmap:
                frame = self._add_heatmap_to_frame(frame, name, i, sigma=self.sigma)

            video_writer.add_frame(frame)

        # 完成视频写入并返回相关信息
        video_writer.release_writer()

        return (video_writer.output_file,
                ''.join(self.data_manager['info'][idx]['label_word']),
                self.data_manager['info'][idx])

    def _add_keypoints_to_frame(self, frame, name, frame_idx):
        """
        在帧上添加关键点。

        参数:
        - frame: 要添加关键点的图像帧。
        - name: 数据的名称。
        - frame_idx: 当前帧的索引。
        """
        for kp in self.keypoints_data[name]['keypoints'][frame_idx]:
            x, y, confidence = kp
            if confidence > self.confidence_threshold and 0 <= x <= frame.shape[1] - 1 and 0 <= y <= frame.shape[0] - 1:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    def _add_heatmap_to_frame(self, frame, name, frame_idx, sigma=10):
        """
        在帧上添加热图。

        参数:
        - frame: 要添加热图的图像帧。
        - name: 数据的名称。
        - frame_idx: 当前帧的索引。
        - sigma: 高斯模糊的sigma参数，默认为10。

        返回:
        - 经过热图处理后的帧。
        """
        heatmap = self.create_heatmap(frame, self.keypoints_data[name]['keypoints'][frame_idx], sigma=sigma)
        alpha = 0.5
        frame = cv2.addWeighted(frame, alpha, heatmap, 1 - alpha, 0)
        return frame

    def create_heatmap(self, image, keypoints, sigma=10):
        """
        根据给定的关键点创建热图。

        参数:
        - image: 原始图像。
        - keypoints: 关键点列表。
        - sigma: 高斯模糊的sigma参数，默认为10。

        返回:
        - 生成的热图。
        """
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        # 准备高斯核
        g = cv2.getGaussianKernel(int(self.sigma * 6), self.sigma)
        g = g * g.T

        for kp in keypoints:
            x, y, confidence = kp
            if confidence > self.confidence_threshold and 0 <= x <= width - 1 and 0 <= y <= height - 1:
                x, y = int(x), int(y)
                x_start = max(0, x - 3 * sigma)
                y_start = max(0, y - 3 * sigma)
                x_end = min(width, x + 3 * sigma)
                y_end = min(height, y + 3 * sigma)
                heatmap[y_start:y_end, x_start:x_end] = np.maximum(
                    heatmap[y_start:y_end, x_start:x_end],
                    g[:y_end - y_start, :x_end - x_start] * confidence)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap


def main(args):
    """
    主函数用于初始化参数解析器，设置命令行参数，并启动可视化器服务。
    """

    # 检查并处理temp-dir参数
    try:
        if args.temp_dir is None:
            # 创建一个临时目录
            temp_dir = tempfile.mkdtemp()
            print(f"Auto-created temporary directory: {temp_dir}")
        else:
            temp_dir = args.temp_dir
    except Exception as e:
        print(f"Failed to create temporary directory: {e}")
        raise

    # 初始化CSL Daily Visualizer实例
    visualizer = CSLDailyVisualizer(
        args.data_dir,
        temp_dir,
        args.server_name,
        args.server_port,
        confidence_threshold=args.confidence_threshold,
        sigma=args.sigma,
        fps=args.fps,
    )

    # 注册一个函数，在程序退出时清理临时目录
    atexit.register(visualizer.temp_dir_manager.clean_tmp_dir)

    # 启动定时清理临时目录的任务，每600秒执行一次
    visualizer.temp_dir_manager.start_repeat_delete(600)

    # 获取数据
    data_len = len(visualizer.data_manager['info'])

    # 初始化Gradio接口
    iface = gr.Interface(
        title='CSL Daily Visualizer',
        fn=visualizer.process_frames_to_video,
        inputs=[
            gr.Number(label='Sample Index', info=f'{0}~{data_len - 1}',
                      maximum=data_len - 1, minimum=0),
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

    # 启动Gradio接口服务
    iface.launch(share=args.share, server_name=args.server_name, server_port=args.server_port, inbrowser=args.browse)


if __name__ == "__main__":
    # 创建一个参数解析器，用于解析命令行参数
    parser = argparse.ArgumentParser(description="CSL Daily Visualizer")

    # 添加必需的命令行参数：数据目录路径
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the data directory")

    # 添加必需的命令行参数：临时目录路径
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Path to the temporary directory or None to auto-create")

    # 添加可选的命令行参数：服务器名称，默认为localhost
    parser.add_argument("--server-name", type=str, default="localhost", help="Server name")

    # 添加可选的命令行参数：服务器端口，默认为28080
    parser.add_argument("--server-port", type=int, default=28080, help="Server port")

    # 添加可选的命令行参数：启动时是否打开浏览器
    parser.add_argument("--browse", action="store_true", help="Open browser on launch")

    # 添加可选的命令行参数：是否共享可视化，默认为False
    parser.add_argument("--share", action="store_false", help="Share the visualization")

    # 添加可选的命令行参数：置信度阈值，默认为0.3
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Confidence threshold for keypoints")

    # 添加可选的命令行参数：高斯核的sigma值，默认为10
    parser.add_argument("--sigma", type=int, default=10, help="Sigma for Gaussian kernel")

    # 添加可选的命令行参数：帧率，默认为25
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用main函数，传入解析后的参数
    main(args)
