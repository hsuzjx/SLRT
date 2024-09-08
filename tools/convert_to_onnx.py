import torch
import argparse
from slr.model import SLRModel


def convert_to_onnx(model, file_path):
    """
    将给定的模型转换为ONNX格式并保存到指定的文件路径。

    参数:
    - model: 需要转换的PyTorch模型。
    - file_path: 保存转换后的ONNX模型的文件路径。

    返回值:
    无
    """
    try:
        # 将模型移动到CPU
        model = model.to('cpu')

        # 设置模型为评估模式
        model.eval()

        # 定义输入样例。这里的样例包括两个输入张量和对应的标签张量。
        input_sample = (
        torch.randn(2, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100, 100]).to('cpu'), torch.randn(100, 2, 1296))

        # 导出模型到ONNX格式。export_params=True表示将模型的参数一起导出。
        torch.onnx.export(model, input_sample, file_path, export_params=True)

    except Exception as e:
        # 捕获并打印可能发生的异常
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format.")

    # 添加模型路径参数
    parser.add_argument("model_path", type=str, help="Path to the PyTorch model file.")

    # 添加输出ONNX文件路径参数
    parser.add_argument("output_path", type=str, help="Output path for the ONNX model file.")

    # 解析命令行参数
    args = parser.parse_args()

    # 加载模型
    pth_model = SLRModel.load_from_checkpoint(args.model_path)

    # 转换模型并保存为ONNX格式
    convert_to_onnx(pth_model, args.output_path)
