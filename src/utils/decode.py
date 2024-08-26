from itertools import groupby

# import ctcdecode
import torch


class Decode(object):
    """
    该类用于将神经网络的输出解码为手语词汇。它支持两种搜索模式：最大值搜索和束搜索。
    
    参数:
    - gloss_dict: 手语词汇字典，将词汇映射到数字。
    - num_classes: 手语词汇的数量。
    - search_mode: 搜索模式，可以是'max'（最大值搜索）或'beam'（束搜索）。
    - beam_width: 束搜索的宽度，默认为10。
    - blank_id: 空白符号的ID，默认为0。
    - num_processes: 并行解码进程的数量，默认为10。
    """

    def __init__(self, gloss_dict, num_classes, search_mode, beam_width=10, blank_id=0, num_processes=10):
        # 初始化反向字典，用于将数字映射回手语词汇
        self.i2g_dict = dict((v, k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id

        # 生成对应数量的字符，用于CTC解码器
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]

        # # 初始化CTC束搜索解码器
        # self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=beam_width,
        #                                             blank_id=blank_id,
        #                                             num_processes=num_processes)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        # """
        # 解码神经网络的输出。
        #
        # 参数:
        # - nn_output: 神经网络的输出，形状为(batch_size, seq_len, num_classes)或(seq_len, batch_size, num_classes)。
        # - vid_lgt: 每个样本的序列长度。
        # - batch_first: 如果为True，输入的维度顺序为(batch_size, seq_len, num_classes)，否则为(seq_len, batch_size, num_classes)。
        # - probs: 如果为True，将输出概率，否则输出对数概率。
        #
        # 返回:
        # - 根据搜索模式，返回相应的解码结果。
        # """
        # # 如果不是批量优先，调整输入维度顺序
        # if not batch_first:
        #     nn_output = nn_output.permute(1, 0, 2)
        #
        # # 根据搜索模式选择解码方法
        # if self.search_mode == "max":
        #     return self.MaxDecode(nn_output, vid_lgt)
        # else:
        #     return self.BeamSearch(nn_output, vid_lgt, probs)
        return None

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        使用束搜索算法对神经网络的输出进行解码。
        
        CTCBeamDecoder的结构：
                - 输入: nn_output (B, T, N)，应通过softmax层
                - 输出: beam_results (B, N_beams, T)，int，需要用i2g_dict解码
                      beam_scores (B, N_beams)，概率值
                      timesteps (B, N_beams)
                      out_lens (B, N_beams)
        
        参数:
        - nn_output: 神经网络的输出，形状为(B, T, N)
        - vid_lgt: 视频长度，形状为(B,)
        - probs: 是否直接使用概率值，默认为False
        
        返回:
        - ret_list: 解码后的文本列表，每个元素为(gloss_id, idx)元组
        '''
        # 如果输入不是概率值，则通过softmax层转换为概率，并移至CPU上处理
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        # 将视频长度移至CPU上处理
        vid_lgt = vid_lgt.cpu()

        # 使用CTC解码器对输入进行解码，获取解码结果、分数、时间步和输出序列长度
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)

        # 初始化结果列表
        ret_list = []
        # 遍历每个样本的解码结果
        for batch_idx in range(len(nn_output)):
            # 获取第一个束的结果，并根据输出序列长度进行截取
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            # 移除连续重复的元素
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            # 将解码结果转换为对应的文本，并添加到结果列表中
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(first_result)])
        # 返回解码后的文本列表
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        """
        对神经网络的输出进行最大值解码。
        
        参数:
        - nn_output: 神经网络的输出，形状为(batchsize, sequence_length, num_classes)。
        - vid_lgt: 视频序列的长度列表，用于指示每个样本的有效长度。
        
        返回值:
        - ret_list: 解码后的结果列表，每个元素包含解码结果和对应的位置索引。
        """
        # 获取每个位置最可能的类别索引
        index_list = torch.argmax(nn_output, axis=2)
        # 获取批次大小和序列长度
        batchsize, lgt = index_list.shape
        # 初始化用于存储解码结果的列表
        ret_list = []

        # 遍历批次中的每个样本
        for batch_idx in range(batchsize):
            # 对每个样本的有效序列部分进行分组处理，去除连续重复的元素
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            # 过滤掉空白标识符
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]

            # 如果存在非空白元素，则进一步处理
            if len(filtered) > 0:
                # 将过滤后的结果转换为tensor，并再次去除连续重复的元素
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                # 如果结果为空，则直接使用过滤后的结果
                max_result = filtered

            # 将解码结果转换为对应的符号，并记录每个符号的位置索引
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(max_result)])

        # 返回解码后的结果列表
        return ret_list
