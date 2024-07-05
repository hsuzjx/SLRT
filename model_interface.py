import os
import sys
import time
import re
import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

import utils
from layers import *
from data_interface import DataInterface
from lightning.pytorch.loggers import WandbLogger

from evaluation.slr_eval.wer_calculation import evaluate


class SLRModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if not os.path.exists(os.path.abspath(self.hparams.gloss_dict_path)):
            os.makedirs(os.path.abspath(self.hparams.gloss_dict_path))
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

        self.decoder = utils.Decode(
            gloss_dict=np.load(os.path.join(os.path.abspath(self.hparams.gloss_dict_path), 'gloss_dict.npy'),
                               allow_pickle=True).item(),
            num_classes=1296, search_mode='beam')

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
        self.total_info += info
        self.total_sentence += pred

        return loss

    def on_validation_epoch_end(self):
        wer = 100.0
        try:
            # sss = open(os.path.join(os.path.abspath(self.hparams.save_path), 'output-hypothesis-dev.ctm'), 'r')
            # aaa = sss.readlines()
            # sss.close()
            self.write2file(os.path.join(os.path.abspath(self.hparams.save_path), 'output-hypothesis-dev.ctm'),
                            self.total_info, self.total_sentence)
            wer = evaluate(mode='dev', sh_path=self.hparams.sh_path,
                           save_path=self.hparams.save_path,
                           ground_truth_path=self.hparams.ground_truth_path,
                           mer_path=self.hparams.mer_path)
        except:
            print("Unexpected error:", sys.exc_info())
            wer = 100.0
        finally:
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            self.log('DEV_WER', wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def write2file(self, path, info, output):
        filereader = open(path, "w")
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                filereader.writelines(
                    "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                     word_idx * 1.0 / 100,
                                                     (word_idx + 1) * 1.0 / 100,
                                                     word[0]))

    def on_test_epoch_start(self):
        self.total_sentence = []
        self.total_info = []

    def test_step(self, batch, batch_idx):
        x, x_lgt, y, y_lgt, info = batch
        y_hat, y_hat_lgt, pred = self(x, x_lgt)

        loss = self.loss_function(y_hat.log_softmax(-1), y.cpu().int(), y_hat_lgt.cpu().int(), y_lgt.cpu().int()).mean()
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)

        # self.validation_step_outputs.append((info, pred_sentence))
        self.total_info += info
        self.total_sentence += pred

        return loss

    def on_test_epoch_end(self):
        wer = 100.0
        try:
            # sss = open(os.path.join(os.path.abspath(self.hparams.save_path), 'output-hypothesis-test.ctm'), 'r')
            # aaa = sss.readlines()
            # sss.close()
            self.write2file(os.path.join(os.path.abspath(self.hparams.save_path), 'output-hypothesis-test.ctm'),
                            self.total_info, self.total_sentence)
            wer = evaluate(mode='test', sh_path=self.hparams.sh_path,
                           save_path=self.hparams.save_path,
                           ground_truth_path=self.hparams.ground_truth_path,
                           mer_path=self.hparams.mer_path)
        except:
            print("Unexpected error:", sys.exc_info())
            wer = 100.0
        finally:
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            self.log('TEST_WER', wer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # log model only if `val_accuracy` increases
    wandb_logger = WandbLogger(project="Phoenix2014", log_model="all",
                               name="TT_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="max")

    data_module = DataInterface(
        features_path='./datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='./datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual',
        gloss_dict_path='./datasets/.tmp',
        ground_truth_path='./datasets/.tmp',
        num_workers=8,
        batch_size=1, )

    model = SLRModel(num_classes=1296, conv_type=2, use_bn=False, hidden_size=1024, gloss_dict=None, weight_norm=True,
                     lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
                     last_epoch=-1,
                     test_param=False, )
    trainer = L.Trainer(max_epochs=3,
                        accelerator='gpu',
                        devices=[0, 1],  # if torch.cuda.is_available() else None,  # limiting got iPython runs
                        # callbacks=[TQDMProgressBar(refresh_rate=20)],
                        # logger=CSVLogger(save_dir="logs/"),
                        profiler="simple",
                        # fast_dev_run=200,
                        limit_train_batches=20,
                        limit_val_batches=10,
                        precision='16-mixed',
                        num_sanity_val_steps=2,
                        log_every_n_steps=2,
                        strategy='ddp_find_unused_parameters_true',
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback],

                        )
    trainer.fit(model,
                datamodule=data_module,
                )

    wandb.finish()
