import os
from datetime import datetime

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, CenterCrop, Normalize

import slr.models
from slr.datasets.tknzs.simple_tokenizer import SimpleTokenizer
from slr.datasets.transforms import ToTensor, TemporalRescale
from slr.utils import convert_to_onnx, set_seed

from slr.models.decoders import CTCBeamSearchDecoder
from slr.evaluation import Evaluator

CONFIG_PATH = '../configs'
CONFIG_NAME = 'CorrNet_Phoenix2014_experiment.yaml'


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    """
    Main function for training and testing a model using Hydra configuration.

    Initializes the training environment, sets up logging, checkpoints, and
    data modules, then trains and tests the model based on the provided config.
    """
    # Set precision for float32 matmul operations
    torch.set_float32_matmul_precision(cfg.get('torch_float32_matmul_precision', 'high'))
    seed = set_seed(cfg.get('seed', -1), workers=True)

    # Define project, name, and timestamp
    project = cfg.get('project', 'default_project')
    name = cfg.get('name', 'default_name')
    timestamp = cfg.get('timestamp', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Create save directory
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize WandbLogger
    wandb.require("core")
    wandb_logger = WandbLogger(
        save_dir=save_dir,
        project=project,
        name=f'{name}_{timestamp}',
        **cfg.logger
    )
    # Update wandb config with random seed
    if isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run):
        wandb_logger.experiment.config.update({'random_seed': seed}, allow_val_change=True)

    # Setup checkpoint callback
    dirpath = os.path.join(save_dir, 'checkpoints')
    os.makedirs(dirpath, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        **cfg.callback
    )

    # Common parameters
    dataset_name = cfg.dataset_name
    model_name = cfg.model_name
    tokenizer = SimpleTokenizer(**cfg.tokenizer)

    ctc_decoder = CTCBeamSearchDecoder(tokenizer=tokenizer, **cfg.decoder)
    evaluator = Evaluator(dataset=dataset_name, **cfg.evaluator)

    # Initialize data module
    DataModelClassDict = {
        "phoenix2014": slr.datasets.Phoenix2014DataModule,
        "phoenix2014T": slr.datasets.Phoenix2014TDataModule,
        "csl-daily": slr.datasets.CSLDailyDataModule
    }
    transform = {
        'train': Compose([ToTensor(), RandomCrop(224), RandomHorizontalFlip(0.5), TemporalRescale(0.2),
                          Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
        'dev': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
        'test': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])])
    }
    data_module = DataModelClassDict[dataset_name](
        transform=transform,
        tokenizer=tokenizer,
        **cfg.dataset
    )

    # Initialize model
    ModelClassDict = {
        "CorrNet": slr.models.CorrNet,
        # "SLRTransformer": slr.models.SLRTransformer,
    }
    eval_res_save_dir = os.path.join(save_dir, 'hypothesis')
    os.makedirs(eval_res_save_dir, exist_ok=True)
    model = ModelClassDict[model_name](
        save_dir=eval_res_save_dir,
        probs_decoder=ctc_decoder,
        evaluator=evaluator,
        **cfg.model
    )

    # Initialize trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **cfg.trainer
    )

    # Train model
    trainer.fit(model, datamodule=data_module)

    # Test the best model
    best_model = getattr(slr.models, model_name).load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    trainer.test(best_model, datamodule=data_module)

    # Ensure wandb.finish() is called
    try:
        wandb.finish()
    except Exception as finish_error:
        print(f"wandb.finish() encountered an issue: {finish_error}")

    # Optionally convert model to ONNX format
    if cfg.get('convert_to_onnx', False):
        best_model = getattr(slr.models, model_name).load_from_checkpoint(checkpoint_callback.best_model_path)
        onnx_save_dir = os.path.join(save_dir, 'onnx')
        os.makedirs(onnx_save_dir, exist_ok=True)
        onnx_file_name = os.path.basename(checkpoint_callback.best_model_path).replace('.ckpt', '.onnx')
        convert_to_onnx(best_model, os.path.join(onnx_save_dir, onnx_file_name))


if __name__ == '__main__':
    main()
