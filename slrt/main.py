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
from slrt.utils import set_seed, set_num_threads

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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
    num_threads = cfg.get('num_threads', psutil.cpu_count(logical=False))
    set_num_threads(num_threads)
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
    dataset_name = cfg.dataset.name
    data_type = cfg.data_type
    model_name = cfg.model.name

    recognition_cfg = cfg.get('recognition', {})
    translation_cfg = cfg.get('translation', {})

    # Create save directory
    save_dir = os.path.join(os.path.abspath(cfg.get('save_dir', '../experiments')), project, name, str(times))
    os.makedirs(save_dir, exist_ok=True)
    # Checkpoint file
    ckpt_file = cfg.get('checkpoint', None)
    # Is convert_to_onnx enabled?
    is_save_onnx = cfg.get('convert_to_onnx', False)

    task = cfg.get('task', ["recognition"])

    #######################################################################################
    ##################### Load Dataset Configurations #####################################
    data_cfgs = cfg.dataset.get('data_cfgs', None)
    kps_file = cfg.dataset.get('keypoints_file', None)

    if "recognition" in task:
        recognition_gt_file = cfg.dataset.get('glosses_groundtruth_file', None)
        recognition_vocab_file = cfg.dataset.get('gloss_vocab_file', None)
    else:
        recognition_gt_file, recognition_vocab_file = None, None

    if "translation" in task:
        translation_gt_file = cfg.dataset.get('translation_groundtruth_file', None)
        translation_vocab_file = cfg.dataset.get('word_vocab_file', None)
    else:
        translation_gt_file, translation_vocab_file = None, None

    #######################################################################################
    ##################### Initialize Tokenizer, Decoder and Evaluator #####################
    # Initialize tokenizer, decoder, and evaluator
    if "recognition" in task:
        recognition_tokenizer = TokenizerDict["Recognition"][recognition_cfg.tokenizer_name](
            vocab_file=recognition_vocab_file, **recognition_cfg.tokenizer)
        recognition_decoder = DecoderDict["Recognition"][recognition_cfg.decoder_name](
            tokenizer=recognition_tokenizer, **recognition_cfg.decoder)
        recognition_evaluator = EvaluatorDict["Recognition"][recognition_cfg.evaluator_name](
            gt_file=recognition_gt_file, dataset=dataset_name, **recognition_cfg.evaluator)
    else:
        recognition_tokenizer, recognition_decoder, recognition_evaluator = None, None, None

    if "translation" in task:
        translation_tokenizer = TokenizerDict["Translation"][translation_cfg.tokenizer_name](
            vocab_file=translation_vocab_file, **translation_cfg.tokenizer)
        translation_decoder = DecoderDict["Translation"][translation_cfg.decoder_name](
            tokenizer=translation_tokenizer, **translation_cfg.decoder)
        translation_evaluator = EvaluatorDict["Translation"][translation_cfg.evaluator_name](
            gt_file=translation_gt_file, dataset=dataset_name, **translation_cfg.evaluator)
    else:
        translation_tokenizer, translation_decoder, translation_evaluator = None, None, None

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
    if data_type == 'video':
        data_module = DataModuleClassDict['video'][dataset_name](
            transform=TransformDict[data_type],
            tokenizer={'recognition': recognition_tokenizer, 'translation': translation_tokenizer},
            **data_cfgs,
            **cfg.dataloader
        )
    elif data_type == 'keypoint':
        data_module = DataModuleClassDict['keypoint'][dataset_name](
            transform=TransformDict[data_type],
            tokenizer={'recognition': recognition_tokenizer, 'translation': translation_tokenizer},
            keypoints_file=kps_file,
            **cfg.dataloader
        )
    elif data_type == 'patch-kps':
        data_module = DataModuleClassDict['patch-kps'][dataset_name](
            transform=TransformDict[data_type],
            tokenizer={'recognition': recognition_tokenizer, 'translation': translation_tokenizer},
            keypoints_file=kps_file,
            **data_cfgs,
            **cfg.dataloader
        )
    else:
        raise ValueError(f"Unsupported dataset type: {data_type}")

    #######################################################################################
    ##################### Initialize Model ################################################
    # Initialize model
    model = ModelClassDict[model_name](
        save_dir=save_dir,
        probs_decoder={'recognition': recognition_decoder, 'translation': translation_decoder},
        evaluator={'recognition': recognition_evaluator, 'translation': translation_evaluator},
        task=task,
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
