import cv2
import os

import cv2
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_video_from_frames_with_annotations(frames_dir, annotations, output_video_path, fps=30,
                                              position=(50, 50),
                                              font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc',
                                              font_size=30,
                                              color=(255, 0, 0),
                                              ):
    """
    :param frames_dir: 包含帧图像的目录
    :param annotations: 一个字符串列表，每个元素对应一个帧的注释
    :param output_video_path: 输出视频的完整路径
    :param fps: 输出视频的帧率，默认为30
    :param position: 一个二元组，定义文本的左上角在图像上的位置 (x, y)。
    :param font_path: 字体文件的路径。
    :param font_size: 文本的字体大小。
    :param color: 一个三元组，定义文本颜色的 RGB 值 (R, G, B)。
    """

    # 获取目录中的所有文件名，假设它们是按顺序排列的
    frame_filenames = sorted(os.listdir(frames_dir))
    assert len(frame_filenames) == len(annotations), "The number of frames and annotations must match."

    # 读取第一张图片以获取尺寸信息
    first_frame = cv2.imread(os.path.join(frames_dir, frame_filenames[0]))
    height, width, _ = first_frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for i, filename in enumerate(frame_filenames):
        frame_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error reading image: {filename}")
            continue

        # 在帧上添加的注释
        text = annotations[i]

        # 将OpenCV图像转换为PIL图像
        pil_frame = Image.fromarray(frame)

        # 创建Draw对象
        draw = ImageDraw.Draw(pil_frame)

        # 设置字体，确保你有中文字体文件或使用系统默认的中文字体
        font = ImageFont.truetype(font_path, font_size)

        # 在图像上添加中文文本
        draw.text(position, text, font=font, fill=color)

        # 将PIL图像转换回OpenCV图像
        frame = np.array(pil_frame)

        # cv2.putText(frame, text, org, font,
        #             fontScale=font_scale, color=color, thickness=thickness)

        # 写入帧到视频
        out.write(frame)

    # 释放资源
    out.release()

# if __name__ == '__main__':
#     import pickle
#
#     with open('/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_label/csl2020ct_v2.pkl', 'rb') as f:
#         data = pickle.load(f)
#     info0 = data['info'][1010]
#
#     # 示例使用
#     frames_directory = os.path.join(
#         '/new_home/xzj23/openmmlab_workspace/SLR/data/csl-daily/sentence_frames-512x512/frames_512x512', info0['name'])
#     annotation_list = ['{}'.format(''.join(info0["label_word"])) for _ in range(info0['length'])]
#     output_directory = f'/new_home/xzj23/openmmlab_workspace/SLR/{info0["name"]}.mp4'
#
#     create_video_from_frames_with_annotations(frames_directory, annotation_list, output_directory)
