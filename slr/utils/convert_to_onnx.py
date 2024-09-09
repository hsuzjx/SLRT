import torch


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
        input_sample = (torch.randn(2, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100, 100]).to('cpu'))

        # 导出模型到ONNX格式。export_params=True表示将模型的参数一起导出。
        model.to_onnx(file_path, input_sample, export_params=True)

    except Exception as e:
        # 捕获并打印可能发生的异常
        print(f"Error occurred: {e}")
