import os.path

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# __all__ = [
#     'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#     'resnet152', 'resnet200'
# ]


# 定义一个字典，用于存储ResNet系列模型的URL地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)


class BasicBlock(nn.Module):
    """
    基本块类，继承自nn.Module。
    该类实现了一个基本的残差网络块，包含两个3x3的卷积层，批量归一化层，以及一个可选的下采样层。
    """
    expansion = 1  # 所有BasicBlock的通道扩张比例均为1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        初始化函数。

        参数:
        inplanes: int，输入通道数。
        planes: int，输出通道数。
        stride: int，卷积步长，默认为1。
        downsample: nn.Module，下采样模块，默认为None。
        """
        super(BasicBlock, self).__init__()

        # 第一个卷积层，使用3x3卷积核，输入通道数为inplanes，输出通道数为planes，卷积步长为stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 第一个批量归一化层，作用于planes通道
        self.bn1 = nn.BatchNorm3d(planes)
        # ReLU激活函数，用于引入非线性特性
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层，使用3x3卷积核，输入通道数和输出通道数均为planes
        self.conv2 = conv3x3(planes, planes)
        # 第二个批量归一化层，作用于planes通道
        self.bn2 = nn.BatchNorm3d(planes)
        # 下采样模块，用于调整维度，使其与主路径的输出匹配，如果需要的话
        self.downsample = downsample
        # 卷积步长，用于后续可能的下采样操作
        self.stride = stride

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x: Tensor，输入张量。

        返回:
        out: Tensor，输出张量。
        """
        # 残差连接，初始输出为输入x
        residual = x

        # 主路径第一部分：卷积、批量归一化、ReLU激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 主路径第二部分：卷积、批量归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，则对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 主路径输出与残差相加，然后通过ReLU激活函数
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet模型类，继承自PyTorch的Module类。

    参数:
    - block: 定义基本块的类型。
    - layers: 每个层包含的基本块数量的列表。
    - num_classes: 分类数，默认为1000。
    """

    def __init__(self, block, layers, num_classes=1000):
        # 初始化输入平面的通道数
        self.inplanes = 64
        # 调用父类的构造函数
        super(ResNet, self).__init__()
        # 定义卷积层，用于模型的初始卷积操作
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        # 定义批归一化层，接在卷积层之后
        self.bn1 = nn.BatchNorm3d(64)
        # 定义ReLU激活函数，用于引入非线性
        self.relu = nn.ReLU(inplace=True)
        # 定义最大池化层，用于下采样
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # 构建模型的第一层
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 构建模型的第二层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建模型的一个层，包含多个基本块。

        参数:
        - block: 基本块类型。
        - planes: 每个基本块的输出通道数。
        - blocks: 该层包含的基本块数量。
        - stride: 步长，用于控制下采样的程度。
        """
        # 定义下采样操作，如果需要改变特征图的尺寸或通道数
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        # 添加基本块到层中
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        定义模型的前向传播过程。

        参数:
        - x: 输入张量，形状为(N, C, T, H, W)。

        返回:
        - x: 输出张量，模型的预测结果。
        """
        # 获取输入的尺寸信息
        N, C, T, H, W = x.size()
        # 初始卷积层
        x = self.conv1(x)
        # 批归一化
        x = self.bn1(x)
        # 激活函数
        x = self.relu(x)
        # 最大池化
        x = self.maxpool(x)

        # 通过模型的第一层
        x = self.layer1(x)
        # 通过模型的第二层
        x = self.layer2(x)


        return x


class ResNet18(ResNet):
    def __init__(self, pretrained=False, model_dir='./.models', num_classes=1000):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes)
        if pretrained:
            model_dir = os.path.abspath(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            # 从预训练模型URL中加载ResNet-18的参数
            checkpoint = model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir)
            # 获取模型参数的层名称列表
            layer_name = list(checkpoint.keys())
            # 遍历层名称列表，对特定层的权重进行维度扩展
            for ln in layer_name:
                if 'conv' in ln or 'downsample.0.weight' in ln:
                    checkpoint[ln] = checkpoint[ln].unsqueeze(2)
            # 加载模型参数，允许非严格模式以忽略不匹配的层
            self.load_state_dict(checkpoint, strict=False)


class ResNet34(ResNet):
    def __init__(self, pretrained=False, model_dir='./.models', num_classes=1000):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes)
        if pretrained:
            model_dir = os.path.abspath(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            checkpoint = model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir)
            layer_name = list(checkpoint.keys())
            for ln in layer_name:
                if 'conv' in ln or 'downsample.0.weight' in ln:
                    checkpoint[ln] = checkpoint[ln].unsqueeze(2)
            # 加载模型参数，允许非严格模式以忽略不匹配的层
            self.load_state_dict(checkpoint, strict=False)
