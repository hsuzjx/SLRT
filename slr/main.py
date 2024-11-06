import os
import shutil
from datetime import datetime

import hydra
import lightning as L
import torch
import transformers
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from slr.constants import DataModuleClassDict, ModelClassDict, transform, CONFIG_PATH, CONFIG_NAME
from slr.datasets.tknzs.simple_tokenizer import SimpleTokenizer
from slr.evaluation import Evaluator
from slr.models.decoders import CTCBeamSearchDecoder
from slr.utils import convert_to_onnx, set_seed


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    """
    Main function for training and testing a model using Hydra configuration.

    Initializes the training environment, sets up logging, checkpoints, and
    data modules, then trains and tests the model based on the provided config.
    """
    # Set num threads
    torch.set_num_threads(cfg.get('torch_num_threads', 2))
    # Set precision for float32 matmul operations
    torch.set_float32_matmul_precision(cfg.get('torch_float32_matmul_precision', 'high'))
    seed = set_seed(cfg.get('seed', -1), workers=True)

    # Define project, name, and times
    project = cfg.get('project', 'default_project')
    name = cfg.get('name', 'default_name')
    times = cfg.get('times', 0)

    # Create save directory
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, str(times))
    os.makedirs(save_dir, exist_ok=True)

    # Common parameters
    dataset_name = cfg.dataset_name
    model_name = cfg.model_name

    # Initialize tokenizer, decoder, and evaluator
    tokenizer = SimpleTokenizer(**cfg.tokenizer)
    # tokenizer = transformers.BertTokenizer.from_pretrained('/new_home/xzj23/workspace/SLR/.cache/huggingface/bert-base-german-dbmdz-uncased')
    ctc_decoder = CTCBeamSearchDecoder(tokenizer=tokenizer, **cfg.decoder)
    evaluator = Evaluator(dataset=dataset_name, **cfg.evaluator)

    # Initialize WandbLogger
    wandb.require("core")
    wandb_logger = WandbLogger(
        save_dir=save_dir,
        project=project,
        name=f'{name}_{times}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        **cfg.logger
    )

    # Initialize data module
    data_module = DataModuleClassDict[dataset_name](
        transform=transform,
        tokenizer=tokenizer,
        **cfg.dataset
    )

    # Initialize model
    model = ModelClassDict[model_name](
        save_dir=save_dir,
        probs_decoder=ctc_decoder,
        evaluator=evaluator,
        **cfg.model
    )

    # Initialize trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        **cfg.trainer
    )

    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.get('checkpoint', None))

    # Copy the best model to the save directory
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model_score = trainer.checkpoint_callback.best_model_score
    shutil.copyfile(best_model_path, os.path.join(os.path.dirname(best_model_path), 'best.ckpt'))
    print("Best Model:", best_model_path, ", Best DEV_WER:", best_model_score.item())

    # Test the best model
    trainer.test(model, datamodule=data_module, ckpt_path=best_model_path)

    # Ensure wandb.finish() is called
    try:
        wandb.finish()
    except Exception as finish_error:
        print(f"wandb.finish() encountered an issue: {finish_error}")

    # Optionally convert model to ONNX format
    if cfg.get('convert_to_onnx', False) and trainer.is_global_zero:
        onnx_save_dir = os.path.join(save_dir, 'onnx')
        os.makedirs(onnx_save_dir, exist_ok=True)
        best_model = ModelClassDict[model_name].load_from_checkpoint(best_model_path)
        best_model.eval()
        convert_to_onnx(best_model, os.path.join(onnx_save_dir, "best.onnx"))


if __name__ == '__main__':
    main()
