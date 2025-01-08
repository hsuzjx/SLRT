import math
import numpy as np
import torch
import torch.nn as nn


class STAttentionModule(nn.Module):
    def __init__(
            self,
            st_attention_module_prams: list[list[int, int, int, int, int]],
            num_channel: int,
            num_node: int,
            max_frame: int,
            num_subset: int = 6,
            glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
            use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1,
            use_pes=True, use_pet=False,
    ):
        super(STAttentionModule, self).__init__()

        self.sta_module_prams = st_attention_module_prams
        assert len(self.sta_module_prams) > 0

        self.in_channel = self.sta_module_prams[0][0]
        self.out_channel = self.sta_module_prams[-1][1]

        self.map_layer = nn.Sequential(
            nn.Conv2d(num_channel, self.in_channel, 1),
            nn.BatchNorm2d(self.in_channel),
            nn.LeakyReLU(0.1),
        )

        self.graph_layers = nn.Sequential()
        num_frame = max_frame
        for index, (in_c, out_c, inter_c, t_k, s) in enumerate(self.sta_module_prams):
            self.graph_layers.add_module(
                f'STAttentionBlock_{index}',
                STAttentionBlock(
                    in_c, out_c, inter_c,
                    stride=s,
                    t_kernel=t_k,
                    num_node=num_node,
                    num_frame=num_frame,
                    num_subset=num_subset,
                    # 其他参数
                    glo_reg_s=glo_reg_s,
                    att_s=att_s,
                    glo_reg_t=glo_reg_t,
                    att_t=att_t,
                    use_temporal_att=use_temporal_att,
                    use_spatial_att=use_spatial_att,
                    attentiondrop=attentiondrop,
                    use_pes=use_pes,
                    use_pet=use_pet,
                )
            )
            num_frame = int(num_frame / s + 0.5)

    def forward(self, x):
        # 获取输入数据的维度
        N, C, T, V = x.shape
        # 调整输入数据的维度顺序并保持连续性
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        x = self.map_layer(x)
        x = self.graph_layers(x)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.mean(3)

        return x


class STAttentionBlock(nn.Module):
    """
    空间-时间注意力块，用于处理输入的特征图，结合空间和时间维度的注意力机制。
    
    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - inter_channels (int): 中间通道数。
    - num_subset (int): 注意力子集的数量，默认为2。
    - num_node (int): 节点的数量，默认为27。
    - num_frame (int): 帧的数量，默认为400。
    - kernel_size (int): 卷积核的大小，默认为1。
    - stride (int): 步长，默认为1。
    - t_kernel (int): 时间维度卷积核的大小，默认为3。
    - glo_reg_s (bool): 是否使用全局空间注意力，默认为True。
    - att_s (bool): 是否使用空间注意力，默认为True。
    - glo_reg_t (bool): 是否使用全局时间注意力，默认为False。
    - att_t (bool): 是否使用时间注意力，默认为False。
    - use_temporal_att (bool): 是否使用时间维度的注意力机制，默认为False。
    - use_spatial_att (bool): 是否使用空间维度的注意力机制，默认为True。
    - attentiondrop (float): 注意力机制中的dropout概率，默认为0。
    - use_pes (bool): 是否使用空间位置编码，默认为True。
    - use_pet (bool): 是否使用时间位置编码，默认为False。
    """

    def __init__(
            self,
            in_channels, out_channels, inter_channels,
            num_subset=2, num_node=27, num_frame=400,
            kernel_size=1, stride=1, t_kernel=3,
            glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
            use_temporal_att=False, use_spatial_att=True, attentiondrop=0.,
            use_pes=True, use_pet=False
    ):
        super(STAttentionBlock, self).__init__()
        # 初始化通道数和注意力机制的参数
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        # 计算卷积的填充大小
        pad = int((kernel_size - 1) / 2)
        # 是否使用空间注意力机制
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            # 初始化空间注意力矩阵和位置编码
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            # 定义空间注意力机制中的前馈网络
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            # 是否使用空间注意力权重
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            # 是否使用全局空间注意力
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            # 定义输出网络
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # 如果不使用空间注意力，直接定义卷积层
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        # 计算时间维度卷积的填充大小
        padd = int(t_kernel / 2)
        # 定义时间维度的卷积层
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        # 根据输入和输出通道数以及步长，决定是否需要下采样
        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # 如果不需要下采样，直接使用恒等映射
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x
        # 定义激活函数和dropout
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        """
        定义前向传播过程，处理输入的张量x并返回处理后的张量z。
        
        参数:
        x (Tensor): 输入的张量，大小为(N, C, T, V)，分别表示批大小、通道数、时间步和节点数。
        
        返回:
        Tensor: 经过网络处理后的输出张量z。
        """
        # 获取输入张量的维度信息
        N, C, T, V = x.size()

        # 判断是否使用空间注意力机制
        if self.use_spatial_att:
            # 初始化注意力矩阵
            attention = self.atts
            # 判断是否使用位置编码
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            # 如果启用了注意力机制
            if self.att_s:
                # 计算查询和键，并计算注意力分数
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv [4,16,t,v]
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            # 如果启用了全局注意力正则化
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            # 应用dropout到注意力矩阵上
            attention = self.drop(attention)
            # 应用注意力矩阵到输入张量上，并进行维度变换
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            # 通过输出网络进行进一步处理
            y = self.out_nets(y)  # nctv
            # 使用残差连接和ReLU激活函数进行非线性变换
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            # 如果不使用空间注意力机制，直接通过输出网络进行处理
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)
        # 进一步的网络处理
        z = self.out_nett(y)
        z = self.relu(self.downt2(y) + z)

        return z


class PositionalEncoding(nn.Module):
    """
    实现位置编码功能。

    位置编码可以分为时间编码和空间编码，取决于domain参数。
    该类的主要作用是为输入数据添加位置信息，以便于后续处理。

    参数:
    - channel: 输入数据的通道数。
    - joint_num: 关节数量。
    - time_len: 时间序列长度。
    - domain: 编码领域，可以是'temporal'（时间）或'spatial'（空间）。
    """

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        # 根据domain参数选择合适的编码方式
        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        # 将位置列表转换为张量，并为其添加一个维度
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # 初始化位置编码张量
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        # 计算位置编码中的分母项
        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        # 计算位置编码的正弦和余弦分量
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码张量重新整形并添加一个批次维度
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        # 将位置编码张量注册为缓冲区变量
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将输入数据与位置编码相加
        x = x + self.pe[:, :, :x.size(2)]
        return x
