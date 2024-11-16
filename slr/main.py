import os
import shutil
from datetime import datetime

import hydra
import lightning as L
import psutil
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
    #######################################################################################
    # Set num threads
    torch_num_threads = cfg.get('torch_num_threads', psutil.cpu_count(logical=False))
    torch.set_num_threads(torch_num_threads)
    # Set precision for float32 matmul operations
    torch_float32_matmul_precision = cfg.get('torch_float32_matmul_precision', 'high')
    torch.set_float32_matmul_precision(torch_float32_matmul_precision)
    # Set seed
    seed = cfg.get('seed', -1)
    set_seed(seed, workers=True)

    #######################################################################################
    # Define project, name, and times
    project = cfg.get('project', 'default_project')
    name = cfg.get('name', 'default_name')
    times = cfg.get('times', 0)
    # Common parameters
    dataset_name = cfg.dataset_name
    dataset_type = cfg.dataset_type
    model_name = cfg.model_name
    # Create save directory
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, str(times))
    os.makedirs(save_dir, exist_ok=True)
    #
    ckpt_file = cfg.get('checkpoint', None)
    # Is convert_to_onnx enabled?
    is_save_onnx = cfg.get('convert_to_onnx', False)

    #######################################################################################
    data_cfgs = cfg.dataset.get('data_cfgs', None)
    kps_file = cfg.dataset.get('keypoints_file', None)
    gloss_gt_file = cfg.dataset.get('gloss_groundtruth_file', None)
    gloss_vocab_file = cfg.dataset.get('gloss_vocab_file', None)

    #######################################################################################
    # Initialize tokenizer, decoder, and evaluator
    tokenizer = SimpleTokenizer(vocab_file=gloss_vocab_file, **cfg.tokenizer)
    # tokenizer = transformers.BertTokenizer.from_pretrained('/new_home/xzj23/workspace/SLR/.cache/huggingface/bert-base-german-dbmdz-uncased')
    ctc_decoder = CTCBeamSearchDecoder(tokenizer=tokenizer, **cfg.decoder)
    evaluator = Evaluator(gt_file=gloss_gt_file, dataset=dataset_name, **cfg.evaluator)

    #######################################################################################
    # Initialize WandbLogger
    wandb.require("core")
    wandb_logger = WandbLogger(
        save_dir=save_dir,
        project=project,
        name=f'{name}_{times}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        **cfg.logger
    )

    #######################################################################################
    # Initialize data module
    if dataset_type == 'video':
        data_module = DataModuleClassDict[dataset_name](
            transform=transform,
            tokenizer=tokenizer,
            **data_cfgs,
            **cfg.dataloader
        )
    elif dataset_type == 'keypoint':
        data_module = DataModuleClassDict[dataset_name](
            transform=transform,
            tokenizer=tokenizer,
            keypoints_file=kps_file,
            **cfg.dataloader
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    #######################################################################################
    # Initialize model
    model = ModelClassDict[model_name](
        save_dir=save_dir,
        probs_decoder=ctc_decoder,
        evaluator=evaluator,
        **cfg.model
    )

    #######################################################################################
    # Initialize trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        **cfg.trainer
    )

    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_file)

    # Get the path and score of the best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model_score = trainer.checkpoint_callback.best_model_score
    print("Best Model:", best_model_path, ", Best DEV_WER:", best_model_score.item())

    # Copy the best model to the save directory
    shutil.copyfile(best_model_path, os.path.join(os.path.dirname(best_model_path), 'best.ckpt'))

    # Test the best model
    trainer.test(model, datamodule=data_module, ckpt_path=best_model_path)

    #######################################################################################
    # Ensure wandb.finish() is called
    try:
        wandb.finish()
    except Exception as finish_error:
        print(f"wandb.finish() encountered an issue: {finish_error}")

    #######################################################################################
    # Optionally convert model to ONNX format
    if is_save_onnx and trainer.is_global_zero:
        onnx_save_dir = os.path.join(save_dir, 'onnx')
        os.makedirs(onnx_save_dir, exist_ok=True)
        best_model = ModelClassDict[model_name].load_from_checkpoint(best_model_path)
        best_model.eval()
        convert_to_onnx(best_model, os.path.join(onnx_save_dir, "best.onnx"))


if __name__ == '__main__':
    main()
