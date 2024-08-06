import atexit
import os
import shutil
import tempfile
import threading

import cv2
import gradio as gr
import pandas as pd
from translate import Translator

TEMP_DIR = '../.tmp'
SERVER_NAME = '10.12.44.154'
SERVER_PORT = 7868

# 定义数据路径
data_path = '/new_home/xzj23/workspace/SLR/data/phoenix2014'
features_path = os.path.join(data_path, 'phoenix-2014-multisigner/features/fullFrame-210x260px')
annotations_path = os.path.join(data_path, 'phoenix-2014-multisigner/annotations/manual')

# 初始化数据列表
features_list = []
annotations_list = []
num_of_samples = dict()

# 加载数据
for mode in ['train', 'dev', 'test']:
    with open(os.path.join(annotations_path, f'{mode}.corpus.csv'), 'rb') as f:
        data = pd.read_csv(f, sep='|')
    for i, info in data.iterrows():
        frames_file_list = sorted(os.listdir(os.path.join(features_path, mode, info['id'], '1')))
        features_list.append(
            [os.path.join(features_path, mode, info['id'], '1', frame_file) for frame_file in frames_file_list])
        annotations_list.append(info)
    num_of_samples[mode] = len(data)


# 定义视频写入器类
class VideoWriter:
    def __init__(self, output_path, fps=25):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = None
        self.writer = None

    def init_writer(self, frame):
        if self.writer is not None:
            return
        self.frame_size = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

    def add_frame(self, frame):
        if self.writer is None:
            self.init_writer(frame)
        self.writer.write(frame)

    def release_writer(self):
        if self.writer is not None:
            self.writer.release()


# 定义函数：将图像序列转换为视频
def frames_to_video(frames_file_list, video_save_file, is_add_key_points=False, keypoints_list=None,
                    is_add_heatmap=False, heatmaps_list=None, fps=25):
    if is_add_key_points and keypoints_list is None:
        raise ValueError("keypoints_list must be provided if is_add_key_points is True")
    if is_add_heatmap and heatmaps_list is None:
        raise ValueError("heatmaps_list must be provided if is_add_heatmap is True")

    video_writer = VideoWriter(video_save_file, fps)

    for frame_file in frames_file_list:
        frame = cv2.imread(frame_file)
        if frame is None:
            raise FileNotFoundError(f"Frame file {frame_file} not found.")

        # TODO: 添加关键点和热力图

        video_writer.add_frame(frame)

    video_writer.release_writer()


# 处理函数，用于Gradio界面
def process_frames_to_video(idx, is_add_keypoints=False, is_add_heatmap=False):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    output_video_path = os.path.join(temp_dir, "output_video.mp4")
    frames_to_video(frames_file_list=features_list[idx], video_save_file=output_video_path, fps=25)

    return output_video_path, annotations_list[idx]['annotation'], annotations_list[idx]


def clean_tmp_dir():
    """
    删除临时目录。
    """
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"Folder {TEMP_DIR} has been deleted.")
    except FileNotFoundError:
        print(f"The folder {TEMP_DIR} does not exist.")
    except PermissionError:
        print(f"Permission denied when trying to delete {TEMP_DIR}.")
    except Exception as e:
        print(f"An error occurred while deleting {TEMP_DIR}: {e}")


def repeat_delete(interval):
    """
    定时删除临时目录。
    """
    clean_tmp_dir()
    timer = threading.Timer(interval, repeat_delete, args=[interval])
    timer.start()


# 注册退出钩子，确保程序退出时清理临时目录
atexit.register(clean_tmp_dir)

if __name__ == "__main__":
    # 每10分钟清理一次临时目录
    repeat_delete(600)
    # 加载数据并启动gradio界面
    iface = gr.Interface(
        title='Phoenix2014 Visualizer',
        # description='Visualize Phoenix2014 dataset.',
        fn=process_frames_to_video,
        inputs=[gr.Number(label='sample index',
                          info=f'train set: {0}~{num_of_samples["train"] - 1}, '
                               f'dev set: {num_of_samples["train"]}~{num_of_samples["train"] + num_of_samples["dev"] - 1}, '
                               f'test set: {num_of_samples["train"] + num_of_samples["dev"]}~{num_of_samples["train"] + num_of_samples["dev"] + num_of_samples["test"] - 1}',
                          maximum=len(features_list) - 1,
                          minimum=0),
                gr.Checkbox(label='is add keypoints?'),
                gr.Checkbox(label='is add heatmap?')],
        outputs=[gr.Video(label='video', autoplay=True, show_download_button=False),
                 gr.Text(label='annotation'),
                 gr.Text(label='info')],
        live=False,
        allow_flagging='never',
    )
    iface.launch(share=False, server_name=SERVER_NAME, server_port=SERVER_PORT)
