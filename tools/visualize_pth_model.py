import argparse
import tempfile

from tools.convert_to_onnx import convert_to_onnx
from tools.visualize_onnx_model import visualize_onnx_model

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Visualize an ONNX model using Netron.")

    # 添加命令行参数
    parser.add_argument("--pth_file", type=str, required=True, help="Path to the PyTorch model file.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--host", type=str, default="localhost", help="Host address for Netron server.")
    parser.add_argument("--port", type=int, default=28080, help="Port number for Netron server.")
    parser.add_argument("--browse", action="store_true", help="Open the model in the default web browser.")

    # 解析命令行参数
    args = parser.parse_args()

    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
            onnx_file = temp_file.name

            # 将.pth文件转换为.onnx文件
            convert_to_onnx(args.model_name, args.pth_file, onnx_file)

            # 可视化ONNX模型
            visualize_onnx_model(onnx_file, args.host, args.port, args.browse)

    except Exception as e:
        print(f"An error occurred: {e}")
