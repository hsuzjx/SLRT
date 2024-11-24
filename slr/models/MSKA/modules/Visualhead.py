import torch
import torch.nn as nn
import torch.nn.functional as F

from slr.models.MSKA.modules.utils import PositionalEncoding, MaskedNorm, PositionwiseFeedForward, MLPHead


class VisualHead(torch.nn.Module):
    """
    VisualHead类用于处理视觉特征并生成字典输出。
    
    参数:
        cls_num: 类别数量。
        input_size: 输入特征的维度，默认为512。
        hidden_size: 隐藏层的维度，默认为1024。
        ff_size: 前馈网络的维度，默认为2048。
        pe: 是否使用位置编码。
        ff_kernelsize: 前馈网络中卷积核的大小。
        pretrained_ckpt: 预训练模型的路径。
        is_empty: 是否为空模型，即不包含任何操作。
        frozen: 是否冻结模型参数。
        plus_conv_cfg: 附加卷积层的配置。
        ssl_projection_cfg: 自监督学习投影层的配置。
    """

    def __init__(
            self,
            cls_num, input_size=512, hidden_size=1024, ff_size=2048, pe=True,
            ff_kernelsize=[3, 3], pretrained_ckpt=None, is_empty=False, frozen=False,
            plus_conv_cfg={},
            ssl_projection_cfg={}
    ):
        super().__init__()

        # 初始化模型参数
        self.is_empty = is_empty
        self.plus_conv_cfg = plus_conv_cfg
        self.ssl_projection_cfg = ssl_projection_cfg

        # 如果模型不为空，则进行以下初始化
        if is_empty == False:
            self.frozen = frozen
            self.hidden_size = hidden_size

            # 根据输入尺寸是否存在，选择身份映射或线性变换
            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)

            # 初始化批量归一化、激活函数和dropout
            self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='batch')
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.1)

            # 初始化位置编码
            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            # 初始化前馈神经网络
            self.feedforward = PositionwiseFeedForward(
                input_size=self.hidden_size,
                ff_size=ff_size,
                dropout=0.1,
                kernel_size=ff_kernelsize,
                skip_connection=True
            )

            # 初始化层归一化
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

            # 根据配置初始化附加卷积层
            if plus_conv_cfg != {}:
                plus_convs = []
                for i in range(plus_conv_cfg['num_layer']):
                    plus_convs.append(
                        nn.Conv1d(
                            self.hidden_size,
                            self.hidden_size,
                            kernel_size=plus_conv_cfg['kernel_size'],
                            stride=plus_conv_cfg['stride'],
                            padding_mode='replicate'
                        )
                    )
                self.plus_conv = nn.Sequential(*plus_convs)
            else:
                self.plus_conv = nn.Identity()

            # 根据配置初始化自监督学习投影层
            if ssl_projection_cfg != {}:
                self.ssl_projection = MLPHead(
                    embedding_size=self.hidden_size,
                    projection_hidden_size=ssl_projection_cfg['hidden_size']
                )

            # 初始化输出层
            self.gloss_output_layer = torch.nn.Linear(self.hidden_size, cls_num)

            # 如果冻结模型，则设置相关层的参数为不可训练
            if self.frozen:
                self.frozen_layers = [self.fc1, self.bn1, self.relu1, self.pe, self.dropout1, self.feedforward,
                                      self.layer_norm]
                for layer in self.frozen_layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                    layer.eval()

        else:
            # 如果模型为空，则直接初始化输出层
            self.gloss_output_layer = torch.nn.Linear(input_size, cls_num)

        # 如果提供预训练模型路径，则加载预训练模型
        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        """
        从预训练模型中加载参数。
        
        参数:
            pretrained_ckpt: 预训练模型的路径。
        """
        logger = get_logger()
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k, v in checkpoint.items():
            if 'recognition_network.visual_head.' in k:
                load_dict[k.replace('recognition_network.visual_head.', '')] = v
        self.load_state_dict(load_dict)
        logger.info('Load Visual Head from pretrained ckpt {}'.format(pretrained_ckpt))

    def forward(self, x, mask, valid_len_in=None):
        """
        前向传播函数。
        
        参数:
            x: 输入特征。
            mask: 掩码。
            valid_len_in: 有效长度。
            
        返回:
            字典，包含多个输出特征和概率。
        """
        # 输入特征的批量大小、时间步长和特征维度
        B, Tin, D = x.shape
        if self.is_empty == False:
            if not self.frozen:
                # 投影层1
                x = self.fc1(x)
                x = self.bn1(x, mask)
                x = self.relu1(x)
                # 位置编码
                x = self.pe(x)
                x = self.dropout1(x)

                # 前馈网络
                x = self.feedforward(x)
                x = self.layer_norm(x)

                # 准备进行卷积操作
                x = x.transpose(1, 2)
                x = self.plus_conv(x)
                x = x.transpose(1, 2)
            else:
                # 如果模型被冻结，则在推理时不计算梯度
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        if ii == 1:
                            x = layer(x, mask)
                        else:
                            x = layer(x)
                # 准备进行卷积操作
                x = x.transpose(1, 2)
                x = self.plus_conv(x)
                x = x.transpose(1, 2)

        # 分类层
        logits = self.gloss_output_layer(x)  # B,T,V
        gloss_probabilities_log = logits.log_softmax(2)
        gloss_probabilities = logits.softmax(2)

        # 计算输出的有效长度
        if self.plus_conv_cfg != {}:
            B, Tout, D = x.shape
            valid_len_out = torch.floor(valid_len_in * Tout / Tin).long()  # B,
        else:
            valid_len_out = valid_len_in
        # SSL投影层
        if self.ssl_projection_cfg != {}:
            x_ssl = self.ssl_projection(x)
            if self.ssl_projection_cfg['normalize'] == True:
                x_ssl = F.normalize(x_ssl, dim=-1)
        else:
            x_ssl = None
        # 返回包含多个输出特征和概率的字典
        return {
            'gloss_feature_ssl': x_ssl,
            'gloss_feature': x,
            'gloss_feature_norm': F.normalize(x, dim=-1),
            'gloss_logits': logits,
            'gloss_probabilities_log': gloss_probabilities_log,
            'gloss_probabilities': gloss_probabilities,
            'valid_len_out': valid_len_out
        }
