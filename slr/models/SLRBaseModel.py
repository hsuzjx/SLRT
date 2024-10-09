import itertools
import os
import re
from datetime import datetime
from typing import Any, List, Tuple

import lightning as L
import torch

from slr.evaluation.wer_calculation import evaluate
from slr.models.decoders import CTCBeamSearchDecoder


class SLRBaseModel(L.LightningModule):
    """
    Base LightningModule for Sign Language Recognition models.

    This class provides common functionalities such as logging metrics, handling gradients,
    and evaluating model performance during training and testing phases.
    """

    def __init__(self, **kwargs):
        """
        Initializes the base model with hyperparameters and sets up necessary components.

        Args:
            kwargs: Hyperparameters needed for model initialization.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Ensure save path exists
        os.makedirs(os.path.abspath(self.hparams.save_dir), exist_ok=True)

        # Initialize CTC decoder
        self.probs_decoder = CTCBeamSearchDecoder(
            tokenizer=self.hparams.tokenizer,
            beam_width=self.hparams.beam_width,
            num_processes=self.hparams.num_processes
        )

        # Initialize outputs lists
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Register hook for handling NaN gradients
        self.register_full_backward_hook(self.handle_nan_gradients)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        loss, _, _, _ = self.step_forward(batch)

        # Log learning rate
        self.log(
            'Train/Learning Rate', self.trainer.optimizers[0].param_groups[0]['lr'],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # Log training loss
        self.log(
            'Train/Loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        # Perform forward pass and compute loss, predictions, and additional info
        loss, y_hat, y_hat_lgt, info = self.step_forward(batch)

        # Log the validation loss
        self.log(
            'Val/Loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # Decode the predictions
        decoded = self.probs_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        # Remove special tokens from the decoded predictions
        for tokens in decoded:
            for token in self.probs_decoder.tokenizer.special_tokens:
                if token == self.probs_decoder.tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Collect the current batch's information and predictions
        self.validation_step_outputs.append({
            'predictions': [(info[i], decoded[i]) for i in range(len(info))]
        })

        # Return the loss value
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value.
        """
        # Perform forward pass and compute loss, predictions, and additional info
        loss, y_hat, y_hat_lgt, info = self.step_forward(batch)

        # Log test loss
        self.log(
            'Test/Loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # Decode the predictions
        decoded = self.probs_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        # Remove special tokens from the decoded predictions
        for tokens in decoded:
            for token in self.probs_decoder.tokenizer.special_tokens:
                if token == self.probs_decoder.tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Collect the current batch's information and predictions
        self.test_step_outputs.append({
            'predictions': [(info[i], decoded[i]) for i in range(len(info))]
        })

        # Return the loss value
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Performs prediction on a batch of data.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.
            dataloader_idx: Index of the dataloader.

        Returns:
            Decoded predictions.
        """
        _, y_hat, y_hat_lgt, info = self.step_forward(batch)
        decoded = self.probs_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        for tokens in decoded:
            for token in self.probs_decoder.tokenizer.special_tokens:
                if token == self.probs_decoder.tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Additional logic can be added here to process predictions, such as saving to a file or returning specific formats.
        # Example: Convert predictions to a more readable form or directly return predictions.

        return decoded

    def on_validation_epoch_start(self):
        """
        Called at the start of each validation epoch.
        """
        self.validation_step_outputs = []

    def on_test_epoch_start(self):
        """
        Called at the start of each test epoch.
        """
        self.test_step_outputs = []

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        # Gather data from all GPUs
        all_validation_step_outputs = [None for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(all_validation_step_outputs, self.validation_step_outputs)
        # Ensure all processes have finished gathering data
        torch.distributed.barrier()

        # Ensure all collected data is not None
        for item in all_validation_step_outputs:
            assert item is not None

        # TODO: 检查是否为主进程
        # if self.trainer.is_global_zero:

        # Merge collected data into a single list
        all_items = list(itertools.chain.from_iterable(all_validation_step_outputs))
        total_predictions = list(itertools.chain.from_iterable(item['predictions'] for item in all_items))

        # Check for duplicate names
        total_names = [name for name, _ in total_predictions]
        assert len(total_names) == len(set(total_names))

        try:
            # Prepare save path and output file
            if self.trainer.sanity_checking:
                file_save_dir = os.path.join(self.hparams.save_dir, "dev", "sanity_check")
            else:
                file_save_dir = os.path.join(self.hparams.save_dir, "dev", f"epoch_{self.current_epoch}")
            if not os.path.exists(file_save_dir):
                os.makedirs(file_save_dir, exist_ok=True)
            output_file = os.path.join(file_save_dir, f'output-hypothesis-dev-rank{self.trainer.global_rank}.ctm')

            # Write predictions to file and compute WER
            self.write2file(output_file, total_predictions)

            # Call evaluate function to compute WER
            wer = evaluate(
                ctm_file=output_file,
                gt_file=os.path.join(
                    self.hparams.ground_truth_dir,
                    f"{self.hparams.dataset_name}-groundtruth-dev_sorted.stm"),
                save_dir=file_save_dir,
                sclite_bin=self.hparams.sclite_bin,
                dataset=self.hparams.dataset_name,
                cleanup=self.hparams.cleanup
            )
        except Exception as e:
            # Handle exceptions and log error information
            print(f"Exception occurred at the end of validation epoch: {e}, please check detailed error message.")
            wer = '100.0'
        finally:
            # Process WER logging, ensuring even string values are logged correctly
            if isinstance(wer, str):
                wer = float(re.findall("\d+\.?\d*", wer)[0])
            # Log DEV_WER metric
            self.log('Val/Word Error Rate', wer, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            # Print different messages based on whether it's a sanity check
            if self.trainer.sanity_checking:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sanity Check, DEV_WER: {wer}%")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {self.current_epoch}, DEV_WER: {wer}%")

    def on_test_epoch_end(self):
        """
        Called at the end of each test epoch.
        """
        # Gather data from all GPUs
        all_test_step_outputs = [None for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather_object(all_test_step_outputs, self.test_step_outputs)
        # Ensure all processes have finished gathering data
        torch.distributed.barrier()

        # Ensure all collected data is not None
        for item in all_test_step_outputs:
            assert item is not None

        # TODO: 检查是否为主进程
        # if self.trainer.is_global_zero:

        # Merge collected data into a single list
        all_items = list(itertools.chain.from_iterable(all_test_step_outputs))
        total_predictions = list(itertools.chain.from_iterable(item['predictions'] for item in all_items))

        # Check for duplicate names
        total_names = [name for name, _ in total_predictions]
        assert len(total_names) == len(set(total_names)), "Duplicate names found in predictions."

        try:
            # Prepare save path and output file
            file_save_dir = os.path.join(self.hparams.save_dir, "test", f"test_after_epoch_{self.current_epoch - 1}")
            if not os.path.exists(file_save_dir):
                os.makedirs(file_save_dir, exist_ok=True)
            output_file = os.path.join(file_save_dir, f'output-hypothesis-test-rank{self.trainer.global_rank}.ctm')

            # Write predictions to file and compute WER
            self.write2file(output_file, total_predictions)

            # Call evaluate function to compute WER
            wer = evaluate(
                ctm_file=output_file,
                gt_file=os.path.join(self.hparams.ground_truth_dir,
                                     f"{self.hparams.dataset_name}-groundtruth-test_sorted.stm"),
                save_dir=file_save_dir,
                sclite_bin=self.hparams.sclite_bin,
                dataset=self.hparams.dataset_name,
                cleanup=self.hparams.cleanup
            )
        except Exception as e:
            # Handle exceptions and log error information
            print(f"An exception occurred at the end of the test epoch: {e}. Please check the detailed error message.")
            wer = '100.0'
        finally:
            # Process WER logging, ensuring even string values are logged correctly
            if isinstance(wer, str):
                wer = float(re.findall(r"\d+\.?\d*", wer)[0])
            # Log TEST_WER metric
            self.log('Test/Word Error Rate', wer, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            # Print messages
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Test after epoch {self.current_epoch - 1}, TEST_WER: {wer}%")

    def write2file(self, path: str, preds_info: List[Tuple[str, List[Tuple[str, int]]]]):
        """
        Writes predictions to a file.

        Args:
            path: Path to the output file.
            preds_info: List of tuples containing the name and predicted words.
        """
        contents = []
        for name, preds in preds_info:
            for word, word_idx in preds:
                line = f"{name} 1 {word_idx * 1.0 / 100:.2f} {(word_idx + 1) * 1.0 / 100:.2f} {word}\n"
                contents.append(line)
        content = "".join(contents)

        try:
            with open(path, "w") as file:
                file.write(content)
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
            # Consider logging to a log file
            # ...

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            A dictionary containing the optimizer and learning rate scheduler.
        """
        try:
            # Retrieve hyperparameters
            learning_rate = self.hparams.lr
            weight_decay = self.hparams.weight_decay
            lr_scheduler_milestones = self.hparams.lr_scheduler_milestones
            lr_scheduler_gamma = self.hparams.lr_scheduler_gamma
            last_epoch = getattr(self.hparams, 'last_epoch', -1)  # Default value
        except AttributeError as e:
            # Raise an error if required hyperparameters are missing
            raise ValueError(f"Missing required hparam: {e}")

        # Initialize the Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_scheduler_milestones,
            gamma=lr_scheduler_gamma,
            last_epoch=last_epoch
        )

        # Return the optimizer and learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    @staticmethod
    def handle_nan_gradients(module, grad_input, grad_output):
        """
        Logs any NaN values found in gradients.

        Args:
            module: The module for which backward is called.
            grad_input: Gradients w.r.t. the inputs.
            grad_output: Gradients w.r.t. the outputs.
        """
        for index, gradient in enumerate(grad_input):
            if gradient is not None and torch.isnan(gradient).any():
                print(f"Warning: NaN values detected in gradient {index}.")
        return grad_input

    def on_after_backward(self) -> None:
        """
        Checks for parameters without gradients after backpropagation.
        """
        if self.hparams.test_param:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print(f"Parameter without gradient: {name}")
