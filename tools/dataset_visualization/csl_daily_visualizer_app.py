import atexit
import os
import pickle
import shutil
import tempfile
import threading
from glob import glob

import cv2
import gradio as gr
import numpy as np

# 读取配置文件，获取数据目录、临时目录和服务器地址
config_path = '../configs/visualizer_config.txt'
server_config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            server_config[key] = value

DATA_DIR = server_config.get('data_dir', '../../data/csl-daily')
TEMP_DIR = server_config.get('temp_dir', '../../.tmp')
SERVER_NAME = server_config.get('server_name', '10.12.44.154')
SERVER_PORT = server_config.get('server_port', 7866)


def safe_pickle_load(file_path):
    """
    安全地加载pickle文件。
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # 确保加载的数据是字典类型
        if not isinstance(data, dict):
            raise ValueError("Invalid data structure")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}


def create_heatmap(image, keypoints, sigma=10):
    """
    根据给定的关键点生成热力图。
    """
    height, width = image.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for kp in keypoints:
        x, y, confidence = kp
        if confidence > 0.3 and 0 <= x <= width - 1 and 0 <= y <= height - 1:
            x, y = int(x), int(y)
            g = cv2.getGaussianKernel(int(sigma * 6), sigma)
            g = g * g.T

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


def process_frames_to_video(idx, is_add_keypoints=False, is_add_heatmap=False):
    """
    将图像序列处理为视频。
    """
    keypoints_file = os.path.join(DATA_DIR, 'csl-daily-keypoints.pkl')
    label_file = os.path.join(DATA_DIR, 'sentence_label', 'csl2020ct_v2.pkl')
    frames_dir = os.path.join(DATA_DIR, 'sentence_frames-512x512', 'frames_512x512')

    data = safe_pickle_load(label_file)
    keypoints_data = safe_pickle_load(keypoints_file)

    if 'info' not in data or idx < 0 or idx >= len(data['info']):
        raise ValueError("Invalid data or index")

    name = data['info'][idx]['name']
    frame_paths = sorted(glob(os.path.join(frames_dir, name, '*.jpg')))
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    output_video_path = os.path.join(temp_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(path)
        if frame is None:
            continue  # 跳过无法读取的帧

        if is_add_keypoints:
            for kp in keypoints_data[name]['keypoints'][i]:
                x, y, confidence = kp
                if confidence > 0.3 and 0 <= x <= width - 1 and 0 <= y <= height - 1:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        if is_add_heatmap:
            kp = keypoints_data[name]['keypoints'][i]
            heatmap = create_heatmap(frame, kp, sigma=20)
            alpha = 0.5
            frame = cv2.addWeighted(frame, alpha, heatmap, 1 - alpha, 0)

        out.write(frame)

    out.release()
    return output_video_path, ''.join(data['info'][idx]['label_word']), data['info'][idx]


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
    data = safe_pickle_load(os.path.join(DATA_DIR, 'sentence_label', 'csl2020ct_v2.pkl'))
    iface = gr.Interface(
        title='CSL Daily Visualizer',
        # description='Visualize CSL Daily videos with keypoints and heatmaps.',
        fn=process_frames_to_video,
        inputs=[gr.Number(label='sample index', info=f'{0}~{len(data["info"]) - 1}',
                          maximum=len(data['info']) - 1,
                          minimum=0),
                gr.Checkbox(label='is add keypoints?'),
                gr.Checkbox(label='is add heatmap?')],
        outputs=[gr.Video(label='video', autoplay=True, show_download_button=False),
                 gr.Text(label='translation'),
                 gr.Text(label='info')],
        live=False,
        allow_flagging='never',
    )
    iface.launch(share=True, server_name=SERVER_NAME, server_port=SERVER_PORT)
