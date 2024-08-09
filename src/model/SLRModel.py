import os
import re
import sys

import lightning as L
import torch

from src.evaluation import evaluate
from src.model.modules import resnet18, Identity, TemporalConv, NormLinear, BiLSTMLayer
from src.utils import Decode


class SLRModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if not os.path.exists(os.path.abspath(self.hparams.save_path)):
            os.makedirs(os.path.abspath(self.hparams.save_path))

        self.loss_function = torch.nn.CTCLoss(reduction='none', zero_infinity=False)

        # self.conv2d = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.conv2d = resnet18()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512, hidden_size=self.hparams.hidden_size,
                                   conv_type=self.hparams.conv_type, use_bn=self.hparams.use_bn,
                                   num_classes=self.hparams.num_classes)
        self.conv1d.fc = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)

        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=self.hparams.hidden_size,
                                          hidden_size=self.hparams.hidden_size, num_layers=2, bidirectional=True)

        # self.classifier = torch.nn.Linear(self.hparams.hidden_size, self.hparams.num_classes)
        self.classifier = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)

        self.decoder = Decode(
            gloss_dict=self.hparams.gloss_dict,
            num_classes=self.hparams.num_classes, search_mode='beam')

        self.pred = None
        # self.validation_step_outputs = []

        self.total_sentence = []
        self.total_info = []
        self.register_backward_hook(self.backward_hook)

    def forward(self, x, lgt):
        batch, temp, channel, height, width = x.shape
        x = self.conv2d(x.permute(0, 2, 1, 3, 4)).view(batch, temp, -1).permute(0, 2, 1)

        rst = self.conv1d(x, lgt)
        x = rst['visual_feat']
        lgt = rst['feat_len']

        rst = self.temporal_model(x, lgt)
        x = rst['predictions']

        outputs = self.classifier(x)
        # pred = None if self.training \
        #     else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        # return outputs, lgt, pred
        return outputs, lgt, pred

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def training_step(self, batch, batch_idx):
        x, x_lgt, y, y_lgt, info = batch
        y_hat, y_hat_lgt, _ = self(x, x_lgt)

        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        # assert not torch.isnan(loss).any()
        # if torch.isnan(loss):
        #     print('\ny_hat.log_softmax(-1):',y_hat.log_softmax(-1),
        #           '\ny.cpu().int():', y.cpu().int(),
        #           '\ny_hat_lgt.cpu().', y_hat_lgt.cpu(),
        #           '\ny_lgt.cpu()', y_lgt.cpu())

        return loss

    # def on_train_epoch_end(self):
    #     all_preds = torch.stack(self.training_step_outputs)
    #     self.log('Epoch Train Loss', all_preds.mean())
    #     self.training_step_outputs.clear()  # free memory

    # def write2file(self, path, info, output):
    #     filereader = open(path, "w")
    #     for sample_idx, sample in enumerate(output):
    #         for word_idx, word in enumerate(sample):
    #             filereader.writelines(
    #                 "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
    #                                                  word_idx * 1.0 / 100,
    #                                                  (word_idx + 1) * 1.0 / 100,
    #                                                  word[0]))

    # def on_validation_epoch_start(self) -> None:
    #     preds = []
    #     labels = []
    def on_validation_epoch_start(self):
        self.total_sentence = []
        self.total_info = []

    def validation_step(self, batch, batch_idx):
        x, x_lgt, y, y_lgt, info = batch
        y_hat, y_hat_lgt, pred = self(x, x_lgt)

        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        # self.validation_step_outputs.append((info, pred_sentence))
        self.total_info += [it.name for it in info]
        self.total_sentence += pred

        return loss

    def on_validation_epoch_end(self):
        wer = 100.0
        try:
            # sss = open(os.path.join(os.path.abspath(self.hparams.save_path), 'output-hypothesis-dev.ctm'), 'r')
            # aaa = sss.readlines()
            # sss.close()
            file_save_path = os.path.join(self.hparams.save_path, "dev", f"epoch_{self.current_epoch}")
            if not os.path.isdir(file_save_path):
                os.makedirs(file_save_path)
            output_file = os.path.join(file_save_path, 'output-hypothesis-dev.ctm')
            self.write2file(output_file, self.total_info, self.total_sentence)
            # wer = evaluate(mode='dev', sh_path=self.hparams.sh_path,
            #                save_path=self.hparams.save_path,
            #                ground_truth_path=self.hparams.ground_truth_path,
            #                mer_path=self.hparams.mer_path)
            wer = evaluate(file_save_path=file_save_path,
                           groundtruth_file=os.path.join(self.hparams.ground_truth_path,
                                                         f"{self.hparams.dataset_name}-groundtruth-dev_sorted.stm"),
                           ctm_file=output_file, evaluate_dir=self.hparams.evaluation_sh_path,
                           sclite_path=self.hparams.evaluation_sclite_path)
        except:
            print("Unexpected error:", sys.exc_info())
            wer = 100.0
        finally:
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            self.log('DEV_WER', wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            print("DEV_WER:", wer)

    def write2file(self, path, info, output):
        contents = []
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
            # 可以考虑将错误记录到日志文件

    def on_test_epoch_start(self):
        self.total_sentence = []
        self.total_info = []

    def test_step(self, batch, batch_idx):
        x, x_lgt, y, y_lgt, info = batch
        y_hat, y_hat_lgt, pred = self(x, x_lgt)

        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        # self.validation_step_outputs.append((info, pred_sentence))
        self.total_info += [it.name for it in info]
        self.total_sentence += pred

        return loss

    def on_test_epoch_end(self):
        wer = '100.0'  # 默认值设为字符串方便后续转换
        try:
            file_save_path = os.path.join(self.hparams.save_path, "test", f"epoch_{self.current_epoch}")
            if not os.path.isdir(file_save_path):
                os.makedirs(file_save_path)
            output_file = os.path.join(file_save_path, 'output-hypothesis-test.ctm')
            self.write2file(output_file, self.total_info, self.total_sentence)

            # 假设evaluate函数已正确实现并返回WER指标
            # wer = evaluate(mode='test', sh_path=self.hparams.sh_path,
            #                save_path=save_path,
            #                ground_truth_path=self.hparams.ground_truth_path,
            #                mer_path=self.hparams.mer_path)

            wer = evaluate(file_save_path=file_save_path,
                           groundtruth_file=os.path.join(self.hparams.ground_truth_path,
                                                         f"{self.hparams.dataset_name}-groundtruth-test_sorted.stm"),
                           ctm_file=output_file, evaluate_dir=self.hparams.evaluation_sh_path,
                           sclite_path=self.hparams.evaluation_sclite_path)
        except Exception as e:  # 捕获更具体的异常，提供更多信息
            print(f"在测试阶段结束时发生异常: {e}, 请检查详细错误信息。")
            wer = '100.0'
        finally:
            wer = float(re.findall("\d+\.?\d*", wer)[0])
            self.log('TEST_WER', wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            print("TEST_WER:", wer)

    # def on_validation_epoch_end(self):
    #     lstm_ret = 100.0
    #     work_dir = './work_dir'
    #     mode = 'dev'
    #     total_info = [i[0] for i in self.validation_step_outputs]
    #     total_sent = [i[1] for i in self.validation_step_outputs]
    #     evaluate_tool = 'sclite'
    #
    #     try:
    #         python_eval = True if evaluate_tool == "python" else False
    #         write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
    #         # write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
    #         #            total_conv_sent)
    #
    #         # conv_ret = evaluate(
    #         #     prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
    #         #     evaluate_dir=cfg.dataset_info['evaluation_dir'],
    #         #     evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
    #         #     output_dir="epoch_{}_result/".format(epoch),
    #         #     python_evaluate=python_eval,
    #         # )
    #         lstm_ret = evaluate(
    #             prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
    #             evaluate_dir='./evaluation/slr_eval',
    #             evaluate_prefix='phoenix2014-groundtruth',
    #             output_dir="epoch_{}_result/".format(self.current_epoch),
    #             python_evaluate=python_eval,
    #             triplet=True,
    #         )
    #     except:
    #         print("Unexpected error:", sys.exc_info()[0])
    #         lstm_ret = 100.0
    #     finally:
    #         print('DEV WER: {}'.format(lstm_ret))
    #         self.log('DEV WER', lstm_ret, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #         # self.validation_step_outputs.clear()
    #         self.validation_step_outputs = []

    # def test_step(self, batch, batch_idx):
    #     x, x_lgt, y, y_lgt, info = batch
    #     y_hat, y_hat_lgt = self(x, x_lgt)
    #
    #     loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
    #     self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
    #     return loss

    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.lr_scheduler_milestones,
            gamma=self.hparams.lr_scheduler_gamma,
            last_epoch=self.hparams.last_epoch,
            # verbose=True,
        )

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler},

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
                    print(name)
