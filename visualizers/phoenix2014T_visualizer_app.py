import os
import tempfile

import pyttsx3
import cv2
import gradio
import pandas as pd
from gradio import Interface
from translate import Translator

data_path = '/new_home/xzj23/openmmlab_workspace/SLR/data/phoenix2014T'
features_path = os.path.join(data_path, 'PHOENIX-2014-T/features/fullFrame-210x260px')
annotations_path = os.path.join(data_path, 'PHOENIX-2014-T/annotations/manual')
features_list = []
annotations_list = []
for mode in ['train', 'dev', 'test']:
    with open(
            os.path.join(annotations_path, f'PHOENIX-2014-T.{mode}.corpus.csv'),
            'rb') as f:
        data = pd.read_csv(f, sep='|')
    for i, info in data.iterrows():
        frames_file_list = sorted(os.listdir(os.path.join(features_path, mode, info['name'])))
        features_list.append(
            [os.path.join(features_path, mode, info['name'], frames_file) for frames_file in frames_file_list])
        annotations_list.append(info)


def frames_to_video(frames_file_list, video_save_file, is_add_key_points=False, keypoints_list=None,
                    is_add_heatmap=False, heatmaps_list=None, fps=25):
    """
    将一系列图像帧文件转换为视频，可选地添加关键点或热力图。

    :param frames_file_list: 图像帧文件路径的列表
    :param video_save_file: 视频保存路径
    :param is_add_key_points: 是否添加关键点，默认为False
    :param keypoints_list: 关键点列表，每个元素对应一个帧的关键点，形状为 (N, 3)，N是关键点数量
    :param is_add_heatmap: 是否添加热力图，默认为False
    :param heatmaps_list: 热力图列表，每个元素对应一个帧的热力图，形状为 (H, W)，H和W分别是高度和宽度
    :param fps: 视频帧率，默认为25
    """

    # 确保传入了关键点或热力图列表
    if is_add_key_points and keypoints_list is None:
        raise ValueError("keypoints_list must be provided if is_add_key_points is True")
    if is_add_heatmap and heatmaps_list is None:
        raise ValueError("heatmaps_list must be provided if is_add_heatmap is True")

    # 读取第一个帧以获取尺寸信息
    frame = cv2.imread(frames_file_list[0])
    height, width, _ = frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_save_file, fourcc, fps, (width, height))

    # 循环处理每一个帧
    for i, frame_file in enumerate(frames_file_list):
        frame = cv2.imread(frame_file)

        # # 添加关键点
        # if is_add_key_points:
        #     for kp in keypoints_list[i]:
        #         x, y, conf = kp
        #         if conf > 0:  # 只绘制置信度大于零的关键点
        #             cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        #
        # # 添加热力图
        # if is_add_heatmap:
        #     heatmap = heatmaps_list[i]
        #     heatmap = cv2.resize(heatmap, (width, height))
        #     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap / np.max(heatmap)), cv2.COLORMAP_JET)
        #     frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

        # 写入帧
        video_writer.write(frame)

    # 释放资源
    video_writer.release()


def process_frames_to_video(idx, is_add_keypoints=False, is_add_heatmap=False):
    if not os.path.exists('/new_home/xzj23/openmmlab_workspace/SLR/.tmp'):
        os.makedirs('/new_home/xzj23/openmmlab_workspace/SLR/.tmp')
    # 创建临时文件夹来保存输出视频
    temp_dir = tempfile.mkdtemp(dir='/new_home/xzj23/openmmlab_workspace/SLR/.tmp')

    output_video_path = os.path.join(temp_dir, "output_video.mp4")
    output_audio_path = os.path.join(temp_dir, "output_audio.wav")

    frames_to_video(frames_file_list=features_list[idx], video_save_file=output_video_path, fps=25)
    translation_en = Translator(from_lang="de", to_lang="en").translate(annotations_list[idx]['translation'])
    translation_zh_cn = Translator(from_lang="de", to_lang="zh-cn").translate(annotations_list[idx]['translation'])
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    # engine.setProperty('voice', 'Mandarin')
    # engine.setProperty('voice', 'english')
    engine.setProperty('voice', 'german')
    engine.save_to_file(annotations_list[idx]['translation'], output_audio_path)
    engine.runAndWait()
    engine.stop()

    return (output_video_path, annotations_list[idx]['translation'],
            output_audio_path,
            translation_en, translation_zh_cn,
            annotations_list[idx])


if __name__ == "__main__":
    # repeat_delete(600)

    # 创建 Gradio 接口
    iface = Interface(
        fn=process_frames_to_video,
        inputs=[gradio.Number(label='sample index', info=f'{0}~{len(features_list) - 1}',
                              maximum=len(features_list) - 1,
                              minimum=0),
                gradio.Checkbox(label='is add keypoints?'),
                gradio.Checkbox(label='is add heatmap?')],
        outputs=[gradio.Video(label='video', autoplay=True, show_download_button=False),
                 gradio.Text(label='translation de'),
                 gradio.Audio(label='audio', autoplay=True, show_download_button=False),
                 gradio.Text(label='translation en'),
                 gradio.Text(label='translation zh-cn'),
                 gradio.Text(label='info')],
        live=False,
        allow_flagging='never',
    )

    # 启动 Gradio 应用
    iface.launch(share=False, server_name='10.12.44.154', server_port=7861)
