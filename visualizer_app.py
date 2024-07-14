import atexit
import shutil
import threading

import cv2
import tempfile
import os
from glob import glob
from gradio import Interface
import gradio

import pickle


def process_frames_to_video(idx):
    with open('/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_label/csl2020ct_v2.pkl', 'rb') as f:
        data = pickle.load(f)

    # 读取第一帧以获取尺寸信息
    frames_dir = os.path.join(
        '/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_frames-512x512/frames_512x512',
        data['info'][idx]['name'])
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

    for path in frame_paths:
        # 读取帧
        frame = cv2.imread(os.path.join(frames_dir, path))

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
        inputs=gradio.Number(label='sample index', info=f'{0}~{len(data["info"]) - 1}',
                             maximum=len(data['info']) - 1,
                             minimum=0),
        outputs=[gradio.Video(label='video', autoplay=True), gradio.Text(label='translation'),
                 gradio.Text(label='info')],
        live=False,
    )

    # 启动 Gradio 应用
    iface.launch(share=True, server_name='10.12.44.154')
