import itertools
import os
import re
from datetime import datetime
from typing import Any

import lightning as L
import torch
from torch.sparse import softmax

from slr.evaluation import evaluate

from torchaudio.models.decoder import ctc_decoder


class SLRBaseModel(L.LightningModule):
    """
    手语识别模型，继承自PyTorch Lightning的LightningModule。
    """

    def __init__(self, **kwargs):
        """
        初始化模型参数和组件。
        """
        super().__init__()  # 调用父类的初始化方法
        self.save_hyperparameters()  # 保存超参数

        # 确保保存目录存在
        os.makedirs(os.path.abspath(self.hparams.save_path), exist_ok=True)

        # 初始化模型验证和测试阶段的输出容器
        self.validation_step_outputs = None
        self.test_step_outputs = None

        # 注册后向传播钩子
        self.register_full_backward_hook(self.handle_nan_gradients)

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
        loss, _, _, _ = self.step_forward(batch)

        # 日志记录
        self.log(
            'lr', self.trainer.optimizers[0].param_groups[0]['lr'],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            'train_loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def on_validation_epoch_start(self):
        """
        在每个验证周期开始时重置句子和信息列表。
        """
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """
        执行单个验证步骤，其中包含计算模型的损失和收集预测结果。

        参数:
        - batch: 当前批次的数据，包含输入和目标
        - batch_idx: 当前批次的索引

        返回:
        - loss: 当前批次的损失值
        """
        loss, y_hat, y_hat_lgt, info = self.step_forward(batch)

        # 记录当前批次的损失，以便后续分析
        self.log(
            'val_loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        beam_search_result = self.hparams.probs_decoder(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        decoded = beam_search_result[:][0].words
        # TODO: 确定是否需要移除连续相同的元素

        # 收集当前批次的信息和预测结果
        self.validation_step_outputs.append({
            'predictions': [(info[i], beam_search_result[i][0].words) for i in range(len(info))]
        })

        return loss

    def on_validation_epoch_end(self):
        """
        在每个验证周期结束时执行特定操作。

        主要功能包括：
        - 从所有GPU上收集数据并合并。
        - 根据训练器的状态准备保存路径。
        - 将预测结果写入文件并计算WER（字错率）。
        - 记录DEV_WER指标。
        """
        # 收集所有GPU上的数据
        all_validation_step_outputs = [None for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(all_validation_step_outputs, self.validation_step_outputs)
        # 确保所有进程完成数据收集
        torch.distributed.barrier()

        # 确保所有收集的数据都不是None
        for item in all_validation_step_outputs:
            assert item is not None

        # TODO: 检查是否为主进程
        # if self.trainer.is_global_zero:

        # 将收集到的数据合并成一个列表
        all_items = list(itertools.chain.from_iterable(all_validation_step_outputs))  # 合并列表
        total_predictions = list(itertools.chain.from_iterable(item['predictions'] for item in all_items))

        # 检查是否有重复的name
        total_names = [name for name, _ in total_predictions]
        assert len(total_names) == len(set(total_names))

        try:
            # 准备保存路径和输出文件
            if self.trainer.sanity_checking:
                file_save_path = os.path.join(self.hparams.save_path, "dev", "sanity_check")
            else:
                file_save_path = os.path.join(self.hparams.save_path, "dev", f"epoch_{self.current_epoch}")
            if not os.path.exists(file_save_path):  # 使用更安全的方式检查路径
                os.makedirs(file_save_path, exist_ok=True)  # 添加 exist_ok 参数避免异常
            output_file = os.path.join(file_save_path, f'output-hypothesis-dev-rank{self.trainer.global_rank}.ctm')

            # 写入预测结果到文件并计算WER
            self.write2file(output_file, total_predictions)

            # 调用evaluate函数计算WER
            wer = evaluate(
                dataset_name=self.hparams.dataset_name,
                file_save_path=file_save_path,
                ground_truth_file=os.path.join(
                    self.hparams.ground_truth_path,
                    f"{self.hparams.dataset_name}-groundtruth-dev_sorted.stm"),
                ctm_file=output_file,
                sclite_path=self.hparams.evaluation_sclite_path,
                remove_tmp_file=self.hparams.remove_eval_tmp_file
            )
        except Exception as e:
            # 异常处理，记录错误信息并设置WER为默认值
            print(f"在验证阶段结束时发生异常: {e}, 请检查详细错误信息。")
            wer = '100.0'
        finally:
            # 处理WER记录，确保即使是字符串形式也能正确记录
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            # 记录DEV_WER指标
            self.log('DEV_WER', wer, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            # 根据是否为sanity check打印不同信息
            if self.trainer.sanity_checking:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sanity Check, DEV_WER: {wer}%")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {self.current_epoch}, DEV_WER: {wer}%")

    def on_test_epoch_start(self):
        """
        在测试阶段开始时重置统计信息。
        """
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        """
        执行单个测试步骤，即处理一个批次的数据。

        参数:
        - batch: 一个批次的数据，包含输入和目标等。
        - batch_idx: 批次的索引。

        返回:
        - loss: 该批次的损失值。
        """
        loss, decoded, info = self.step_forward(batch)

        # 记录损失
        self.log(
            'test_loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # 收集当前批次的信息和预测结果
        assert len(info) == len(decoded)
        self.test_step_outputs.append({
            'predictions': [(info[i].name, decoded[i]) for i in range(len(info))]
        })

        return loss

    def on_test_epoch_end(self):
        """
        在测试阶段结束时进行汇总和计算WER（词错误率）。
        """
        # 收集所有GPU上的数据
        all_test_step_outputs = [None for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(all_test_step_outputs, self.test_step_outputs)
        # 确保所有进程完成数据收集
        torch.distributed.barrier()

        # 确保所有收集的数据都不是None
        for item in all_test_step_outputs:
            assert item is not None

        # TODO: 检查是否为主进程
        # if self.trainer.is_global_zero:

        # 将收集到的数据合并成一个列表
        all_items = list(itertools.chain.from_iterable(all_test_step_outputs))  # 合并列表
        total_predictions = list(itertools.chain.from_iterable(item['predictions'] for item in all_items))

        # 检查是否有重复的name
        total_names = [name for name, _ in total_predictions]
        assert len(total_names) == len(set(total_names))

        try:
            # 构造保存路径并创建目录
            file_save_path = os.path.join(self.hparams.save_path, "test",
                                          f"test_after_epoch_{self.current_epoch - 1}")
            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path, exist_ok=True)
            # 定义输出文件路径
            output_file = os.path.join(file_save_path, f'output-hypothesis-test-rank{self.trainer.global_rank}.ctm')

            # 将预测结果写入文件
            self.write2file(output_file, total_predictions)

            # 调用evaluate函数计算WER
            wer = evaluate(
                dataset_name=self.hparams.dataset_name,
                file_save_path=file_save_path,
                ground_truth_file=os.path.join(
                    self.hparams.ground_truth_path,
                    f"{self.hparams.dataset_name}-groundtruth-test_sorted.stm"),
                ctm_file=output_file,
                sclite_path=self.hparams.evaluation_sclite_path,
                remove_tmp_file=self.hparams.remove_eval_tmp_file
            )
        except Exception as e:  # 捕获更具体的异常，提供更多信息
            print(f"在测试阶段结束时发生异常: {e}, 请检查详细错误信息。")
            wer = '100.0'
        finally:
            # 提取数字部分
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            # 记录和输出TEST_WER
            self.log('TEST_WER', wer, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Test after epoch {self.current_epoch - 1}, TEST_WER: {wer}%")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        执行单个预测步骤，即处理一个批次的数据用于预测。

        参数:
        - batch: 一个批次的数据，包含输入等。
        - batch_idx: 批次的索引。
        - dataloader_idx: 当使用多个数据加载器时，标识特定的数据加载器。

        返回:
        - decoded: 解码后的预测结果。
        """
        _, decoded, _ = self.step_forward(batch)

        # 可以在此处添加额外的逻辑来处理预测结果，例如保存到文件、返回特定格式的数据等。
        # 示例: 将预测结果转换为易于理解的形式或直接返回预测结果。

        return decoded

    def write2file(self, path, preds_info):
        """
        将预测结果写入指定文件。

        参数:
        - path: 文件路径
        - info: 附加信息列表
        - output: 预测结果列表
        """
        contents = []
        # 构建文件内容
        for name, preds in preds_info:
            for word, word_idx in preds:
                line = "{} 1 {:.2f} {:.2f} {}\n".format(
                    name,
                    word_idx * 1.0 / 100,
                    (word_idx + 1) * 1.0 / 100,
                    word
                )
                contents.append(line)
        content = "".join(contents)

        try:
            with open(path, "w") as file:
                file.write(content)
        except IOError as e:
            print(f"写入文件时发生错误: {e}")
            # 考虑记录到日志文件
            # ...

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
