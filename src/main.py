import random
import time

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.datamodules import Phoenix2014DataModule
from src.models import SLRModel

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    seed = random.randint(1, 2 ** 32 - 1)
    project = "Phoenix2014"
    name = "TTT-bh-sclite_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

    L.seed_everything(seed, workers=True)

    # log model only if `val_accuracy` increases
    wandb_logger = WandbLogger(project=project,
                               # log_model="all",
                               name=name, offline=True)

    wandb_logger.log_hyperparams({'random_seed': seed})
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/{}_{}'.format(project, name), monitor="DEV_WER",
                                          mode="min", save_last=True, save_top_k=1)

    data_module = Phoenix2014DataModule(
        features_path='../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='../data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
        gloss_dict_path='./.tmp',
        ground_truth_path='./.tmp',
        num_workers=8,
        batch_size=2,
    )

    model = SLRModel(
        num_classes=1296, conv_type=2, use_bn=False, hidden_size=1024, gloss_dict_path='./.tmp',
        save_path='./.tmp/{}_{}'.format(project, name),
        sh_path='src/evaluation/slr_eval',
        ground_truth_path='./.tmp',
        mer_path='src/evaluation/slr_eval',
        weight_norm=True,
        lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
        last_epoch=-1,
        test_param=False,
    )

    trainer = L.Trainer(
        max_epochs=40,
        accelerator='gpu',
        devices=[1],  # if torch.cuda.is_available() else None,  # limiting got iPython runs
        # callbacks=[TQDMProgressBar(refresh_rate=20)],
        # logger=CSVLogger(save_dir="logs/"),
        profiler="simple",
        # fast_dev_run=200,
        # limit_train_batches=20,
        # limit_val_batches=10,
        precision='16-mixed',
        # precision='32',
        num_sanity_val_steps=2,
        log_every_n_steps=2,
        strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )

    # model = model.load_from_checkpoint(
    #     '/new_home/xzj23/openmmlab_workspace/SLR/Phoenix2014/2z2yh3q0/checkpoints/epoch=0-step=2835.ckpt')
    trainer.test(
        model,
        datamodule=data_module,
    )

    wandb.finish()
