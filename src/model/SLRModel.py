import os
import re
from datetime import datetime

import lightning as L
import torch
from torch import nn

from src.evaluation import evaluate
from src.model.modules import resnet18, Identity, TemporalConv, NormLinear, BiLSTMLayer
from src.utils import Decode


class SLRModel(L.LightningModule):
    """
    手语识别模型，继承自PyTorch Lightning的LightningModule。
    """

    def __init__(self, **kwargs):
        """
        初始化模型参数和组件。
        """
        super().__init__()
        self.save_hyperparameters()  # 保存超参数

        # 创建保存路径的父目录
        save_directory = os.path.dirname(os.path.abspath(self.hparams.save_path))
        try:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        except Exception as e:
            print(f"Error creating directory: {e}")

        # 定义损失函数
        self.loss_function = nn.CTCLoss(reduction='none', zero_infinity=False)

        # 初始化卷积层
        self.conv2d = resnet18()
        self.conv2d.fc = Identity()  # 将全连接层替换为身份映射

        # 初始化一维卷积层
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=self.hparams.hidden_size,
            conv_type=self.hparams.conv_type,
            use_bn=self.hparams.use_bn,
            num_classes=self.hparams.num_classes
        )
        self.conv1d.fc = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)  # 定义全连接层

        # 初始化双向 LSTM 层
        self.temporal_model = BiLSTMLayer(
            rnn_type='LSTM',
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=2,
            bidirectional=True
        )

        # 初始化分类器
        self.classifier = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)

        # 初始化解码器
        self.decoder = Decode(
            gloss_dict=self.hparams.gloss_dict,
            num_classes=self.hparams.num_classes,
            search_mode='beam'
        )

        # 注册后向传播钩子
        self.register_full_backward_hook(self.handle_nan_gradients)

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

        # 通过双向 LSTM 层
        lstm_output = self.temporal_model(visual_features, feature_lengths)
        predictions = lstm_output['predictions']

        # 通过分类器
        output_logits = self.classifier(predictions)

        # 解码
        decoded = [] if self.training else self.decoder.decode(output_logits, feature_lengths, batch_first=False,
                                                               probs=False)
        return output_logits, feature_lengths, decoded

    def handle_nan_gradients(self, module, grad_input, grad_output):
        """
        捕获并处理含有 NaN 值的梯度。
        """
        for index, gradient in enumerate(grad_input):
            if gradient is not None and torch.isnan(gradient).any():
                print(f"NaN values detected in gradient {index}.")
        return grad_input

    def training_step(self, batch, batch_idx):
        """
        执行一个训练步骤，包括正向传播、损失计算等。

        参数:
        - batch: 一个批处理的数据，包含输入和目标等。
        - batch_idx: 批处理的索引。

        返回:
        - loss_mean: 该批处理的平均损失。
        """
        # 解包批处理数据
        x, x_lgt, y, y_lgt, info = batch
        # 模型正向传播
        y_hat, y_hat_lgt, _ = self(x, x_lgt)

        # 计算损失
        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int())

        # 检查是否有 NaN 值
        if torch.isnan(loss).any():
            print('\nWARNING:Detected NaN in loss.')
        # 可以添加更详细的调试信息
        # ...
        # 为了程序的健壮性，可以选择跳过该批次或终止训练
        # return None 或者 raise Exception("NaN detected in loss.")

        # 计算平均损失
        loss_mean = loss.mean()

        # 日志记录
        self.log('train_loss', loss_mean, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0],
                 sync_dist=True)

        return loss_mean

    def on_validation_epoch_start(self):
        """
        在每个验证周期开始时重置句子和信息列表。
        """
        self.total_sentence = []
        self.total_info = []

    def validation_step(self, batch, batch_idx):
        """
        执行单个验证步骤，其中包含计算模型的损失和收集预测结果。
    
        参数:
        - batch: 当前批次的数据，包含输入和目标
        - batch_idx: 当前批次的索引
    
        返回:
        - loss: 当前批次的损失值
        """
        # 解包批次数据
        x, x_lgt, y, y_lgt, info = batch
        # 模型前向传播
        y_hat, y_hat_lgt, pred = self(x, x_lgt)

        # 计算并记录验证损失
        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        # 收集当前批次的信息和预测结果
        self.total_info += [it.name for it in info]
        self.total_sentence += pred

        return loss

    def on_validation_epoch_end(self):
        """
        在每个验证周期结束时，计算并记录模型的错误率（WER）。
        """
        wer = '100.0'
        try:
            # 准备保存路径和输出文件
            if self.trainer.sanity_checking:
                file_save_path = os.path.join(self.hparams.save_path, "dev", "sanity_check")
            else:
                file_save_path = os.path.join(self.hparams.save_path, "dev", f"epoch_{self.current_epoch}")
            if not os.path.exists(file_save_path):  # 使用更安全的方式检查路径
                os.makedirs(file_save_path, exist_ok=True)  # 添加 exist_ok 参数避免异常
            output_file = os.path.join(file_save_path, 'output-hypothesis-dev.ctm')

            # 写入预测结果到文件并计算WER
            self.write2file(output_file, self.total_info, self.total_sentence)
            wer = evaluate(file_save_path=file_save_path,
                           groundtruth_file=os.path.join(self.hparams.ground_truth_path,
                                                         f"{self.hparams.dataset_name}-groundtruth-dev_sorted.stm"),
                           ctm_file=output_file, evaluate_dir=self.hparams.evaluation_sh_path,
                           sclite_path=self.hparams.evaluation_sclite_path)
        except Exception as e:
            print(f"在验证阶段结束时发生异常: {e}, 请检查详细错误信息。")
            wer = '100.0'
        finally:
            # 处理WER记录，确保即使是字符串形式也能正确记录
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            self.log('DEV_WER', wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if self.trainer.sanity_checking:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sanity Check, DEV_WER: {wer}%")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {self.current_epoch}, DEV_WER: {wer}%")

    def write2file(self, path, info, output):
        """
        将预测结果写入指定文件。
    
        参数:
        - path: 文件路径
        - info: 附加信息列表
        - output: 预测结果列表
        """
        contents = []
        # 构建文件内容
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                line = "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                        word_idx * 1.0 / 100,
                                                        (word_idx + 1) * 1.0 / 100,
                                                        word[0])
                contents.append(line)
        content = "".join(contents)

        try:
            with open(path, "w") as file:
                file.write(content)
        except IOError as e:
            print(f"写入文件时发生错误: {e}")
            # 考虑记录到日志文件
            # ...

    def on_test_epoch_start(self):
        """
        在测试阶段开始时重置统计信息。
        """
        # 初始化用于存储所有句子和相关信息的列表
        self.total_sentence = []
        self.total_info = []

    def test_step(self, batch, batch_idx):
        """
        执行单个测试步骤，即处理一个批次的数据。
    
        参数:
        - batch: 一个批次的数据，包含输入和目标等。
        - batch_idx: 批次的索引。
    
        返回:
        - loss: 该批次的损失值。
        """
        # 解包批次数据
        x, x_lgt, y, y_lgt, info = batch
        # 模型前向传播
        y_hat, y_hat_lgt, pred = self(x, x_lgt)

        # 计算损失
        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
        # 记录损失
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        # 更新总的信息和句子
        self.total_info += [it.name for it in info]
        self.total_sentence += pred

        return loss

    def on_test_epoch_end(self):
        """
        在测试阶段结束时进行汇总和计算WER（词错误率）。
        """
        # 默认词错误率为100.0，以字符串形式方便后续转换
        wer = '100.0'
        try:
            # 构造保存路径并创建目录
            file_save_path = os.path.join(self.hparams.save_path, "test", f"epoch_{self.current_epoch}")
            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path, exist_ok=True)
            # 定义输出文件路径
            output_file = os.path.join(file_save_path, 'output-hypothesis-test.ctm')
            # 将预测结果写入文件
            self.write2file(output_file, self.total_info, self.total_sentence)

            # 调用evaluate函数计算WER
            wer = evaluate(file_save_path=file_save_path,
                           groundtruth_file=os.path.join(self.hparams.ground_truth_path,
                                                         f"{self.hparams.dataset_name}-groundtruth-test_sorted.stm"),
                           ctm_file=output_file, evaluate_dir=self.hparams.evaluation_sh_path,
                           sclite_path=self.hparams.evaluation_sclite_path)
        except Exception as e:  # 捕获更具体的异常，提供更多信息
            print(f"在测试阶段结束时发生异常: {e}, 请检查详细错误信息。")
            wer = '100.0'
        finally:
            # 提取数字部分
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            # 记录和输出TEST_WER
            self.log('TEST_WER', wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            print("TEST_WER:", wer)

    def configure_optimizers(self):
        # 定义模型优化器
        try:
            # 从模型超参数中获取学习率、权重衰减、学习率调度器的里程碑和伽马值，以及上一个训练周期
            lr = self.hparams.lr
            weight_decay = self.hparams.weight_decay
            lr_scheduler_milestones = self.hparams.lr_scheduler_milestones
            lr_scheduler_gamma = self.hparams.lr_scheduler_gamma
            last_epoch = getattr(self.hparams, 'last_epoch', -1)  # 设置默认值
        except AttributeError as e:
            # 如果模型超参数中缺少必要的属性，抛出ValueError异常
            raise ValueError("Missing required hparam: {}".format(e))

        # 使用Adam优化器初始化优化器，使用从超参数获取的学习率和权重衰减
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # 定义学习率调度器
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,  # 优化器
            milestones=lr_scheduler_milestones,  # 学习率衰减的里程碑
            gamma=lr_scheduler_gamma,  # 学习率衰减的比例
            last_epoch=last_epoch,  # 上一个训练周期
            # verbose=True,  # 可选参数，设置为True以输出调度信息
        )

        # 返回优化器和学习率调度器的字典
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    # def configure_custom_metrics(self):
    #     self.acc = Accuracy()
    #     self.loss = nn.BCEWithLogitsLoss()
    #     self.metrics = {
    #         'acc': self.acc,
    #         'loss': self.loss,
    #         'loss': self.loss,
    #         'loss': self.loss,
    #         'acc': self.acc,
    #     }
    #     return self.metrics

    def on_after_backward(self) -> None:
        if self.hparams.test_param:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print(f"Parameter without gradient: {name}")
