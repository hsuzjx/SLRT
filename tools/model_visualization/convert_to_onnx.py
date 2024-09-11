import argparse

import torch

import slr.model
from tools.INPUT_SAMPLE_DICT import INPUT_SAMPLE


def convert_to_onnx(model_name, pth_file, onnx_file):
    """
    将给定的模型转换为ONNX格式并保存到指定的文件路径。

    参数:
    - model_name: 模型名称。
    - pth_file: PyTorch 模型的检查点文件路径。
    - onnx_file: 保存转换后的ONNX模型的文件路径。

    返回值:
    无
    """
    try:
        # 检查模型名称是否有效
        if model_name not in INPUT_SAMPLE.keys():
            raise ValueError(f"Invalid model name: {model_name}")

        # 加载模型并转移到CPU上进行后续操作
        model = getattr(slr.model, model_name).load_from_checkpoint(pth_file)
        model = model.to('cpu')
        model.eval()  # 将模型设置为评估模式

        # 获取模型对应的输入样本
        input_sample = INPUT_SAMPLE[model_name]

        # 导出模型为ONNX格式并保存
        torch.onnx.export(model, input_sample, onnx_file, export_params=True)

    except Exception as e:
        # 输出异常信息
        print(f"Error occurred: {type(e).__name__} - {e}")


# 当脚本被直接运行时，执行以下代码
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format.")

    # 添加命令行参数
    parser.add_argument("--model_name", type=str, required=True, help="Name of the PyTorch model.")
    parser.add_argument("--pth_file", type=str, required=True, help="Path to the PyTorch checkpoint file.")
    parser.add_argument("--onnx_file", type=str, required=True, help="Path to save the ONNX model.")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行模型转换
    convert_to_onnx(args.model_name, args.pth_file, args.onnx_file)
