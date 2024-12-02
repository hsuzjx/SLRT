import os
import shutil
from datetime import datetime

import hydra
import lightning as L
import psutil
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from slrt.constants import DataModuleClassDict, ModelClassDict, TransformDict, CONFIG_PATH, CONFIG_NAME, TokenizerDict, \
    DecoderDict, EvaluatorDict, InputSampleDict
from slrt.utils import set_seed


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    """
    Main function for training and testing a model using Hydra configuration.

    Initializes the training environment, sets up logging, checkpoints, and
    data modules, then trains and tests the model based on the provided config.
    """
    #######################################################################################
    ##################### Set PyTorch Configurations and Random Seed ######################
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
    ##################### Define Experiment Details #######################################
    # Define project, name, and times
    project = cfg.get('project', 'default_project')
    name = cfg.get('name', 'default_name')
    times = cfg.get('times', 0)
    # Common parameters
    dataset_name = cfg.dataset_name
    dataset_type = cfg.dataset_type
    model_name = cfg.model_name
    tokenizer_name = cfg.tokenizer_name
    decoder_name = cfg.decoder_name
    evaluator_name = cfg.evaluator_name

    # Create save directory
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, str(times))
    os.makedirs(save_dir, exist_ok=True)
    # Checkpoint file
    ckpt_file = cfg.get('checkpoint', None)
    # Is convert_to_onnx enabled?
    is_save_onnx = cfg.get('convert_to_onnx', False)

    #######################################################################################
    ##################### Load Dataset Configurations #####################################
    data_cfgs = cfg.dataset.get('data_cfgs', None)
    kps_file = cfg.dataset.get('keypoints_file', None)
    gt_file = cfg.dataset.get('glosses_groundtruth_file', None)
    vocab_file = cfg.dataset.get('gloss_vocab_file', None)

    #######################################################################################
    ##################### Initialize Tokenizer, Decoder and Evaluator #####################
    # Initialize tokenizer, decoder, and evaluator
    tokenizer = TokenizerDict[tokenizer_name](vocab_file=vocab_file, **cfg.tokenizer)
    ctc_decoder = DecoderDict[decoder_name](tokenizer=tokenizer, **cfg.decoder)
    evaluator = EvaluatorDict[evaluator_name](gt_file=gt_file, dataset=dataset_name, **cfg.evaluator)

    #######################################################################################
    ##################### Initialize WandbLogger ##########################################
    # Initialize WandbLogger
    wandb.require("core")
    wandb_logger = WandbLogger(
        save_dir=save_dir,
        project=project,
        name=f'{name}_{times}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        **cfg.logger
    )

    #######################################################################################
    ##################### Initialize Datamodule ###########################################
    # Initialize data module
    if dataset_type == 'video':
        data_module = DataModuleClassDict[dataset_name](
            transform=TransformDict[dataset_type],
            tokenizer=tokenizer,
            **data_cfgs,
            **cfg.dataloader
        )
    elif dataset_type == 'keypoint':
        data_module = DataModuleClassDict[dataset_name](
            transform=TransformDict[dataset_type],
            tokenizer=tokenizer,
            keypoints_file=kps_file,
            **cfg.dataloader
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    #######################################################################################
    ##################### Initialize Model ################################################
    # Initialize model
    model = ModelClassDict[model_name](
        save_dir=save_dir,
        probs_decoder=ctc_decoder,
        evaluator=evaluator,
        **cfg.model
    )

    #######################################################################################
    ##################### Initialize Trainer; Train and Test ##############################
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
    ##################### Finish W&B ######################################################
    # Ensure wandb.finish() is called
    try:
        wandb.finish()
    except Exception as finish_error:
        print(f"wandb.finish() encountered an issue: {finish_error}")

    #######################################################################################
    ##################### Convert to ONNX #################################################
    # Optionally convert model to ONNX format
    input_sample = InputSampleDict.get(model_name, None)
    if is_save_onnx and trainer.is_global_zero and input_sample:
        onnx_save_dir = os.path.join(save_dir, 'onnx')
        os.makedirs(onnx_save_dir, exist_ok=True)
        best_model = ModelClassDict[model_name].load_from_checkpoint(best_model_path).to('cpu')
        best_model.eval()
        try:
            best_model.to_onnx(
                os.path.join(onnx_save_dir, "best.onnx"),
                input_sample,
                export_params=True,
                opset_version=16
            )
        except Exception as e:
            print(f"Error occurred: {e}")


if __name__ == '__main__':
    main()
