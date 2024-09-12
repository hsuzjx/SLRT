import argparse

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

import slr.model
from slr.model.utils import Decode


class VideoCaptioner:
    """
    视频字幕生成器类，用于为视频生成描述性字幕。
    """

    def __init__(self, model_name, ckpt_file, device):
        """
        初始化VideoCaptioner类。

        :param model_name: 模型名称
        :param ckpt_file: 模型检查点文件路径
        :param device: 设备类型，如'cuda:0'或'cpu'
        """
        self.device = device
        # 通过模型名称和检查点文件路径加载模型，并移动到指定设备
        self.model = getattr(slr.model, model_name).load_from_checkpoint(ckpt_file).to(self.device)

        # 加载gloss字典
        with open("/new_home/xzj23/workspace/SLR/data/global_files/gloss_dict/phoenix2014_gloss_dict.npy", "rb") as f:
            gloss_dict = np.load(f, allow_pickle=True).item()
        # 初始化解码器
        self.decoder = Decode(
            gloss_dict=gloss_dict,
            num_classes=1296,
            search_mode='beam'
        )

    def preprocess_video(self, video_path):
        """
        读取视频文件，并对每一帧进行预处理。

        :param video_path: 视频文件路径
        :return: 处理后的视频帧张量
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise IOError("无法打开视频文件，请检查路径是否正确")

            # 定义变换
            transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 将BGR转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 应用变换
                frame_tensor = transform(Image.fromarray(frame_rgb))
                frames.append(frame_tensor)

            cap.release()

            # 将所有的帧堆叠成一个张量
            frames = torch.stack(frames)
            return frames
        except Exception as e:
            print(f"处理视频时发生错误: {e}")
            raise

    def generate_caption(self, frames):
        """
        根据视频帧生成描述性字幕。

        :param frames: 视频帧张量
        :return: 生成的字幕
        """
        lgt = torch.tensor([frames.shape[0]]).to(self.device)
        # 扩展帧张量
        frames = torch.cat((
            frames[0][None].expand(6, -1, -1, -1),
            frames,
            frames[-1][None].expand(6, -1, -1, -1),
        ), dim=0)
        frames = torch.unsqueeze(frames, 0).to(self.device)

        # 生成文本描述
        with torch.no_grad():
            conv1d_hat, y_hat, y_hat_lgt = self.model(frames, lgt)
            # 解码生成的序列
            decoded = self.decoder.decode(y_hat, y_hat_lgt, batch_first=False, probs=False)
        return decoded


# 创建Gradio接口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="")

    # 添加命令行参数
    parser.add_argument("--model_name", type=str, required=True, default="", help="Name of the model")
    parser.add_argument("--ckpt_file", type=str, required=True, default="", help="Path to the model checkpoint file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device type, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--host", type=str, default="localhost", help="Server address")
    parser.add_argument("--port", type=int, default=28081, help="Server port")
    parser.add_argument("--browse", action="store_true", help="Open the service in the browser automatically")
    parser.add_argument("--share", action="store_true", help="Share the service link")

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化模型
    model = getattr(slr.model, args.model_name).load_from_checkpoint(args.ckpt_file).to(args.device)
    video_captioner = VideoCaptioner(args.model_name, args.ckpt_file, args.device)


    def generate_video_caption(video_path):
        """
        为给定的视频路径生成字幕。

        :param video_path: 视频文件路径
        :return: 生成的字幕文本
        """
        frames = video_captioner.preprocess_video(video_path)
        decoded = video_captioner.generate_caption(frames)
        return " ".join(s[0] for s in decoded[0])


    iface = gr.Interface(
        fn=generate_video_caption,
        inputs=gr.Video(),
        outputs=gr.Text(label='annotation'),
        title="Sign Language Recognition",
        description="",
        live=False,
        allow_flagging='never',
    )

    # 启动Web服务
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=args.browse,
        share=args.share
    )
