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


class Get_Correlation(nn.Module):
    """
    该类用于实现获取特征图之间的相关性。
    
    Attributes:
        channels (int): 输入特征图的通道数。
        reduction_channel (int): 降维后的通道数，为输入通道数的1/16。
        down_conv (nn.Conv3d): 用于降维的3D卷积层。
        down_conv2 (nn.Conv3d): 用于特征变换的3D卷积层。
        spatial_aggregation1 (nn.Conv3d): 用于空间聚合的3D卷积层，使用标准卷积。
        spatial_aggregation2 (nn.Conv3d): 用于空间聚合的3D卷积层，使用扩张卷积。
        spatial_aggregation3 (nn.Conv3d): 用于空间聚合的3D卷积层，使用扩张卷积。
        weights (nn.Parameter): 空间聚合层的权重参数，用于加权求和。
        weights2 (nn.Parameter): 特征融合层的权重参数，用于加权求和。
        conv_back (nn.Conv3d): 用于升维的3D卷积层。
    """

    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        前向传播函数，用于计算输入特征图的相关性。
        
        Parameters:
            x (torch.Tensor): 输入特征图，形状为(B, C, T, H, W)。
            
        Returns:
            torch.Tensor: 输出特征图，形状与输入相同。
        """
        x2 = self.down_conv2(x)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x,
                                  torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2))  # 重复最后一帧
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x,
                                   torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2))  # 重复第一帧
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2),
                                torch.sigmoid(affinities) - 0.5) * self.weights2[0] + \
                   torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2),
                                torch.sigmoid(affinities2) - 0.5) * self.weights2[1]

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x) * self.weights[0] + self.spatial_aggregation2(x) * self.weights[1] \
                       + self.spatial_aggregation3(x) * self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        return features * (torch.sigmoid(aggregated_x) - 0.5)


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
        # 构建相关性模块的第一部分
        self.corr1 = Get_Correlation(self.inplanes)
        # 构建模型的第三层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 构建相关性模块的第二部分
        self.corr2 = Get_Correlation(self.inplanes)
        # 定义参数alpha，用于控制相关性模块的融合程度
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        # 构建模型的第四层
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 构建相关性模块的第三部分
        self.corr3 = Get_Correlation(self.inplanes)
        # 定义平均池化层，用于全局平均池化
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 定义全连接层，用于分类
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        # 加入相关性模块的第一部分
        x = x + self.corr1(x) * self.alpha[0]
        # 通过模型的第三层
        x = self.layer3(x)
        # 加入相关性模块的第二部分
        x = x + self.corr2(x) * self.alpha[1]
        # 通过模型的第四层
        x = self.layer4(x)
        # 加入相关性模块的第三部分
        x = x + self.corr3(x) * self.alpha[2]
        # 调整特征图的维度顺序
        x = x.transpose(1, 2).contiguous()
        # 调整张量形状为(batch_size*T, C, H, W)
        x = x.view((-1,) + x.size()[2:])

        # 全局平均池化
        x = self.avgpool(x)
        # 将张量展平
        x = x.view(x.size(0), -1)
        # 全连接层，得到最终的分类分数
        x = self.fc(x)

        return x


def resnet18(pretrained=False, model_dir='./.models', **kwargs):
    """构建一个基于ResNet-18的模型。
    
    参数:
    **kwargs: 传递给ResNet构造器的额外关键字参数。
    
    返回:
    构建好的ResNet-18模型。
    """
    # 创建ResNet-18模型实例，使用BasicBlock作为基础块，层数配置为[2, 2, 2, 2]
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
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
        model.load_state_dict(checkpoint, strict=False)
    # 返回构建好的模型
    return model

# 
# def resnet34(**kwargs):
#     """构建一个ResNet-34模型。
#     
#     参数:
#     **kwargs: 传递给ResNet构造器的额外关键字参数。
#     
#     返回:
#     构建好的ResNet-34模型。
#     """
#     # 创建ResNet-34模型实例，使用BasicBlock作为基础块，层数配置为[3, 4, 6, 3]
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     # 返回构建好的模型
#     return model

#
# def test():
#     net = resnet18()
#     y = net(torch.randn(1, 3, 224, 224))
#     print(y.size())
#
# # test()
