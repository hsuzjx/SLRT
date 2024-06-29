import time

import torch
import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from data_interface import DataInterface
from model_interface import SLRModel

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # log model only if `val_accuracy` increases
    wandb_logger = WandbLogger(project="Phoenix2014", log_model="all",
                               name="TT_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="max")

    data_module = DataInterface(
        features_path='./datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='./datasets/phoenix2014-release/phoenix-2014-multisigner/annotations/manual',
        gloss_dict_path='./.tmp',
        ground_truth_path='./.tmp',
        num_workers=8,
        batch_size=2, )

    model = SLRModel(num_classes=1296, conv_type=2, use_bn=False, hidden_size=1024, gloss_dict_path='./.tmp',
                     save_path='./.tmp',
                     sh_path='./evaluation/slr_eval',
                     ground_truth_path='./.tmp',
                     mer_path='./evaluation/slr_eval',
                     weight_norm=True,
                     lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
                     last_epoch=-1,
                     test_param=False,

                     )
    trainer = L.Trainer(max_epochs=40,
                        accelerator='gpu',
                        devices=[0],  # if torch.cuda.is_available() else None,  # limiting got iPython runs
                        # callbacks=[TQDMProgressBar(refresh_rate=20)],
                        # logger=CSVLogger(save_dir="logs/"),
                        profiler="simple",
                        # fast_dev_run=200,
                        # limit_train_batches=20,
                        # limit_val_batches=10,
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
    trainer.test(model,
                 datamodule=data_module,
                 )

    wandb.finish()
