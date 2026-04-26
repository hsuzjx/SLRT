import argparse
import os

import netron


def visualize_onnx_model(onnx_file, host, port, browse=False):
    """
    对给定的ONNX模型文件进行可视化。

    参数:
    - onnx_file: ONNX模型文件的路径。
    - host: Netron服务器的主机地址。
    - port: Netron服务器的端口号。
    - browse: 是否在默认浏览器中打开模型。

    异常:
    - 若文件不存在或不是有效文件，抛出 ValueError。
    - 若文件为空，抛出 ValueError。
    - 若文件不是ONNX格式，抛出 ValueError。
    """
    # 验证 onnx_file 是否为合法的文件路径
    if not os.path.exists(onnx_file) or not os.path.isfile(onnx_file):
        raise ValueError(f"The file '{onnx_file}' does not exist or is not a valid file.")

    # 检查文件是否为空
    if os.path.getsize(onnx_file) == 0:
        raise ValueError(f"The file '{onnx_file}' is empty.")

    # 检查文件是否为 .onnx 格式
    if not onnx_file.endswith('.onnx'):
        raise ValueError(f"The file '{onnx_file}' is not an ONNX file.")

    try:
        # 使用 netron 进行模型可视化
        netron.start(onnx_file, address=(host, port), browse=browse)
    except Exception as e:
        # 异常处理
        print(f"An error occurred while visualizing the model: {e}")


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Visualize an ONNX model using Netron.")

    # 添加命令行参数
    parser.add_argument("--onnx-file", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--host", type=str, default="localhost", help="Host address for Netron server.")
    parser.add_argument("--port", type=int, default=28080, help="Port number for Netron server.")
    parser.add_argument("--browse", action="store_true", help="Open the model in the default web browser.")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行模型可视化
    visualize_onnx_model(args.onnx_file, args.host, args.port, args.browse)
