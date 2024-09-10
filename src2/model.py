import torch.nn
import torch_npu

from src.model.modules import TemporalConv, NormLinear, BiLSTMLayer, resnet18, Identity


class SLRModel(torch.nn.Module):
    """
    手语识别模型，继承自PyTorch Lightning的LightningModule。
    """

    def __init__(self, **kwargs):
        """
        初始化模型参数和组件。
        """
        super().__init__()  # 调用父类的初始化方法

        # 初始化网络结构
        self._init_networks()

    def forward(self, inputs, lengths):
        """
        模型的前向传播函数。
        :param inputs: 输入数据，形状为[batch_size, sequence_length, channels, height, width]。
        :param lengths: 输入序列的长度。
        :return: 输出 logits、特征长度和解码结果。
        """
        batch_size, sequence_length, channels, height, width = inputs.shape
        reshaped_inputs = inputs.permute(0, 2, 1, 3, 4)
        # 通过卷积网络
        convolved = self.conv2d(reshaped_inputs).view(batch_size, sequence_length, -1).permute(0, 2, 1)

        # 通过一维卷积层
        conv1d_output = self.conv1d(convolved, lengths)
        visual_features = conv1d_output['visual_feat']
        feature_lengths = conv1d_output['feat_len']
        conv1d_logits = conv1d_output['conv_logits']

        # 通过双向 LSTM 层
        lstm_output = self.temporal_model(visual_features, feature_lengths)
        predictions = lstm_output['predictions']

        # 通过分类器
        output_logits = self.classifier(predictions)

        torch_npu.npu.empty_cache()

        return conv1d_logits, output_logits, feature_lengths

    def _init_networks(self):
        """
        初始化模型的各个网络组件。
        """
        self.conv2d = self._init_conv2d()
        self.conv1d = self._init_conv1d()
        self.temporal_model = self._init_bilstm()

        self.classifier = self.conv1d.fc

    def _init_conv2d(self):
        """
        初始化2D卷积层。
        返回:
            - conv2d: 使用ResNet-18作为2D卷积层，去除其全连接层。
        """
        conv2d = resnet18(pretrained=False)
        conv2d.fc = Identity()  # 将全连接层替换为身份映射
        return conv2d

    def _init_conv1d(self):
        """
        初始化1D卷积层。
        返回:
            - conv1d: 包含全连接层的1D卷积层。
        """
        conv1d = TemporalConv(
            input_size=512,
            hidden_size=1024,
            conv_type=2,
            use_bn=False,
            num_classes=1296
        )
        conv1d.fc = NormLinear(1024, 1296)  # 定义全连接层
        return conv1d

    def _init_bilstm(self):
        """
        初始化双向LSTM层。
        返回:
            - 一个双向LSTM层，用于序列建模。
        """
        return BiLSTMLayer(
            rnn_type='LSTM',
            input_size=1024,
            hidden_size=1024,
            num_layers=2,
            bidirectional=True
        )

    def _init_classifier(self):
        """
        初始化分类器。
        返回:
            - 一个线性分类器，用于将LSTM的输出映射到类别上。
        """
        return NormLinear(1024, 1296)
