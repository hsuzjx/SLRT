import os.path
import random
import time

import lightning as L
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from src.datasets import Phoenix2014DataModule
from src.model import SLRModel
from src.utils import preprocess

if __name__ == '__main__':
    # get config
    with open('../configs/config.yaml', 'r') as f:
        arg = yaml.load(f, Loader=yaml.FullLoader)

    # set precision
    torch.set_float32_matmul_precision(arg.get('global').get('torch_float32_matmul_precision'))

    # set random seed
    seed = arg.get('global').get('seed')
    if seed == -1:
        seed = random.randint(1, 2 ** 32 - 1)
    L.seed_everything(seed, workers=True)

    # set wandb logger
    project = arg.get('wandb').get('project')
    name = arg.get('wandb').get('name').get('name')
    if arg.get('wandb').get('name').get('splice_time'):
        name = "{}_{}".format(name, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    save_dir = os.path.join(arg.get('wandb').get('save_dir'), '{}/{}'.format(project, name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    is_offline = arg.get('wandb').get('offline')
    wandb_logger = WandbLogger(project=project,
                               # log_model="all",
                               name=name, offline=is_offline, save_dir=save_dir, )
    wandb.require("core")

    # log seed
    # wandb_logger.log_hyperparams({'random_seed': seed})
    wandb_logger.experiment.config.update({'seed': seed})

    # set checkpoint
    dirpath = os.path.join(arg.get('checkpoint').get('save_dir'), '{}/{}'.format(project, name), 'checkpoints')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    monitor = arg.get('checkpoint').get('monitor')
    mode = arg.get('checkpoint').get('mode')
    save_last = arg.get('checkpoint').get('save_last')
    save_top_k = arg.get('checkpoint').get('save_top_k')
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                          monitor=monitor,
                                          mode=mode,
                                          save_last=save_last,
                                          save_top_k=save_top_k)

    # preprocess
    dataset_name = arg.get('dataset').get('name')
    gloss_dict_path = arg.get('preprocess').get('gloss_dict_path')
    ground_truth_path = arg.get('preprocess').get('ground_truth_path')
    feature_path = arg.get('dataset').get('feature_path')
    annotations_path = arg.get('dataset').get('annotations_path')
    preprocess(dataset_name=dataset_name,
               annotations_path=annotations_path,
               gloss_dict_path=gloss_dict_path,
               ground_truth_path=ground_truth_path)

    # get gloss dict
    gloss_dict = np.load(os.path.join(gloss_dict_path, '{}_gloss_dict.npy'.format(dataset_name)),
                         allow_pickle=True).item()

    # get lightning data module
    batch_size = arg.get('train').get('batch_size')
    num_workers = arg.get('train').get('num_workers')
    data_module = Phoenix2014DataModule(
        features_path=feature_path,
        annotations_path=annotations_path,
        gloss_dict=gloss_dict,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    # set model
    save_path = os.path.join(arg.get('evaluation').get('save_path'), '{}/{}'.format(project, name), 'hypothesis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sh_path = arg.get('evaluation').get('sh_path')
    mer_path = arg.get('evaluation').get('mer_path')
    model = SLRModel(
        num_classes=1296, conv_type=2, use_bn=False, hidden_size=1024,
        gloss_dict=gloss_dict,
        save_path=save_path,
        sh_path=sh_path,
        ground_truth_path=ground_truth_path,
        mer_path=mer_path,
        weight_norm=True,
        lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
        last_epoch=-1,
        test_param=False,
    )

    # set trainer
    max_epochs = arg.get('train').get('n_epochs')
    accelerator = arg.get('train').get('accelerator')
    devices = arg.get('train').get('devices')
    precision = arg.get('train').get('precision')
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,  # if torch.cuda.is_available() else None,  # limiting got iPython runs
        # callbacks=[TQDMProgressBar(refresh_rate=20)],
        # logger=CSVLogger(save_dir="logs/"),
        profiler="simple",
        # fast_dev_run=200,
        # limit_train_batches=20,
        # limit_val_batches=10,
        precision=precision,
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
