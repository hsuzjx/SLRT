import math
import numpy as np
import torch
import torch.nn as nn


class DSTA(nn.Module):
    """
    定义DSTA网络，用于处理序列数据，如视频中的动作识别。
    
    参数:
    - num_frame: 输入序列的帧数。
    - num_subset: 子图的数量。
    - dropout: Dropout的概率。
    - cfg: 配置文件。
    - args: 额外的参数。
    - num_channel: 输入数据的通道数。
    - glo_reg_s: 是否使用全局空间正则化。
    - att_s: 是否使用空间注意力机制。
    - glo_reg_t: 是否使用全局时间正则化。
    - att_t: 是否使用时间注意力机制。
    - use_temporal_att: 是否使用时间注意力机制。
    - use_spatial_att: 是否使用空间注意力机制。
    - attentiondrop: 注意力机制的dropout概率。
    - use_pet: 是否使用位置编码。
    - use_pes: 是否使用序列位置编码。
    - mode: 模型模式，如'SLR'。
    """

    def __init__(
            self,
            net_prams,
            num_frame=1000,
            num_subset=6,
            dropout=0.1,
            # cfg=None,
            # args=None,
            num_channel=2, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
            use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1, use_pet=False,
            use_pes=True, mode='SLR',
    ):
        super(DSTA, self).__init__()
        self.mode = mode
        # self.cfg = cfg
        # self.args = args
        config = net_prams
        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.num_frame = num_frame
        param = {
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }

        # 定义输入映射，将输入数据转换为适合网络处理的形式
        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        # 定义面部、左手、右手、身体的时空注意力块
        self.face_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.face_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels,
                    stride=stride, t_kernel=t_kernel,
                    num_node=26,
                    num_frame=num_frame,
                    **param
                )
            )
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.left_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.left_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels,
                    stride=stride, num_node=27,
                    t_kernel=t_kernel, num_frame=num_frame,
                    **param
                )
            )
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.right_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.right_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels,
                    stride=stride, t_kernel=t_kernel,
                    num_frame=num_frame,
                    **param
                )
            )
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.body_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.body_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels,
                    stride=stride, num_node=79,
                    t_kernel=t_kernel, num_frame=num_frame,
                    **param
                )
            )
            num_frame = int(num_frame / stride + 0.5)

        self.drop_out = nn.Dropout(dropout)

    def forward(self, kps, body_idx, left_idx, right_idx, face_idx):
        """
        前向传播函数。
        
        参数:
        - kps: 输入的关键点数据。
        
        返回:
        - output: 网络的输出。
        - left_output: 左手的输出。
        - right_output: 右手的输出。
        - body: 身体的输出。
        """
        # 初始化输入数据
        x = kps
        # 获取输入数据的维度
        N, C, T, V = x.shape
        # 调整输入数据的维度顺序并保持连续性
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        # 提取左手、右手、脸部和身体的关键点数据
        left = self.left_input_map(x[:, :, :, left_idx])
        right = self.right_input_map(x[:, :, :, right_idx])
        face = self.face_input_map(x[:, :, :, face_idx])
        body = self.body_input_map(x[:, :, :, body_idx])

        # 通过图卷积层处理脸部、左手、右手和身体的关键点数据
        for i, m in enumerate(self.face_graph_layers):
            face = m(face)
        for i, m in enumerate(self.left_graph_layers):
            left = m(left)
        for i, m in enumerate(self.right_graph_layers):
            right = m(right)
        for i, m in enumerate(self.body_graph_layers):
            body = m(body)  # [B,256,T/4,N] -> [B,256]

        # 调整维度顺序并保持连续性，以进行后续的平均操作
        left = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()
        face = face.permute(0, 2, 1, 3).contiguous()
        body = body.permute(0, 2, 1, 3).contiguous()

        # 在特定维度上计算平均值，以减少数据的维度
        body = body.mean(3)
        face = face.mean(3)
        left = left.mean(3)
        right = right.mean(3)

        # 按照左手、脸部、右手和身体的顺序合并数据
        output = torch.cat([left, face, right, body], dim=-1)
        # 合并左手和脸部数据作为左手的输出
        left_output = torch.cat([left, face], dim=-1)
        # 合并右手和脸部数据作为右手的输出
        right_output = torch.cat([right, face], dim=-1)
        # 返回网络的输出、左手的输出、右手的输出和身体的输出
        return output, left_output, right_output, body


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
