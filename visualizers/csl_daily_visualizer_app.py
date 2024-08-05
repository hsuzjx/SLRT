import atexit
import shutil
import threading

import cv2
import tempfile
import os
from glob import glob

import numpy as np
from gradio import Interface
import gradio

import pickle


def create_heatmap(image, keypoints, sigma=10):
    """
    为给定的关键点生成热力图
    :param image: 原始图像
    :param keypoints: 特征点列表，每个特征点是一个包含位置(x,y)和强度(confidence)的元组
    :param sigma: 高斯滤波器的标准差
    :return: 热力图图像
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
    with open('/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_label/csl2020ct_v2.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/csl-daily-keypoints.pkl', 'rb') as f:
        keypoints = pickle.load(f)
    name = data['info'][idx]['name']
    # 读取第一帧以获取尺寸信息
    frames_dir = os.path.join(
        '/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_frames-512x512/frames_512x512',
        name)
    frame_paths = sorted(os.listdir(frames_dir))
    first_frame = cv2.imread(os.path.join(frames_dir, frame_paths[0]))
    height, width, _ = first_frame.shape

    if not os.path.exists('/new_home/xzj23/openmmlab_workspace/SLR/.tmp'):
        os.makedirs('/new_home/xzj23/openmmlab_workspace/SLR/.tmp')
    # 创建临时文件夹来保存输出视频
    temp_dir = tempfile.mkdtemp(dir='/new_home/xzj23/openmmlab_workspace/SLR/.tmp')

    output_video_path = os.path.join(temp_dir, "output_video.mp4")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))

    for i, path in enumerate(frame_paths):
        # 读取帧
        frame = cv2.imread(os.path.join(frames_dir, path))

        if is_add_keypoints:
            for kp in keypoints[name]['keypoints'][i]:
                x, y, confidence = kp
                if confidence > 0.3 and 0 <= x <= width - 1 and 0 <= y <= height - 1:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)  # 画圆表示关键点
        if is_add_heatmap:
            kp = keypoints[name]['keypoints'][i]
            heatmap = create_heatmap(frame, kp, sigma=20)
            alpha = 0.5
            frame = cv2.addWeighted(frame, alpha, heatmap, 1 - alpha, 0)

        # 将处理后的帧写入输出视频
        out.write(frame)

    # 释放资源
    out.release()

    return output_video_path, ''.join(data['info'][idx]['label_word']), data['info'][idx]


# 删除.tmp
def clean_tmp_dir():
    tmp_dir = '/new_home/xzj23/openmmlab_workspace/SLR/.tmp'
    try:
        shutil.rmtree(tmp_dir)
        print(f"Folder {tmp_dir} has been deleted.")
    except FileNotFoundError:
        print(f"The folder {tmp_dir} does not exist.")
    except PermissionError:
        print(f"Permission denied when trying to delete {tmp_dir}.")
    except Exception as e:
        print(f"An error occurred while deleting {tmp_dir}: {e}")


# 定义一个函数来重复执行删除操作
def repeat_delete(interval):
    clean_tmp_dir()
    timer = threading.Timer(interval, repeat_delete, args=[interval])
    timer.start()


atexit.register(clean_tmp_dir)

if __name__ == "__main__":
    repeat_delete(600)

    with open('/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_label/csl2020ct_v2.pkl', 'rb') as f:
        data = pickle.load(f)
    # 创建 Gradio 接口
    iface = Interface(
        fn=process_frames_to_video,
        inputs=[gradio.Number(label='sample index', info=f'{0}~{len(data["info"]) - 1}',
                              maximum=len(data['info']) - 1,
                              minimum=0),
                gradio.Checkbox(label='is add keypoints?'),
                gradio.Checkbox(label='is add heatmap?')],
        outputs=[gradio.Video(label='video', autoplay=True, show_download_button=False),
                 gradio.Text(label='translation'),
                 gradio.Text(label='info')],
        live=False,
        allow_flagging='never',
    )

    # 启动 Gradio 应用
    iface.launch(share=True, server_name='10.12.44.154')
