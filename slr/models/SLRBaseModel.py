import os
import re
from datetime import datetime
from typing import Any

import fcntl
import lightning as L
import torch


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

        # Initialize variables
        self.file_save_dir = None
        self.output_file = None
        self.lock_file = self.hparams.get("lock_file", "/tmp/slr_file_lock")

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
            'Train/Learning-Rate', self.trainer.optimizers[0].param_groups[0]['lr'],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[-1])
        )

        # Log training loss
        self.log(
            'Train/Loss', loss,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[-1])
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
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[-1])
        )

        # Decode the predictions
        decoded = self.hparams.probs_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        # Remove special tokens from the decoded predictions
        for tokens in decoded:
            for token in self.hparams.probs_decoder.tokenizer.special_tokens:
                if token == self.hparams.probs_decoder.tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Write the decoded predictions to the output file
        for i in range(len(info)):
            self.write_to_file(f"{info[i]} {' '.join(decoded[i])}\n")

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
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[-1])
        )

        # Decode the predictions
        decoded = self.hparams.probs_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        # Remove special tokens from the decoded predictions
        for tokens in decoded:
            for token in self.hparams.probs_decoder.tokenizer.special_tokens:
                if token == self.hparams.probs_decoder.tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Write the decoded predictions to the output file
        for i in range(len(info)):
            self.write_to_file(f"{info[i]} {' '.join(decoded[i])}\n")

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
        decoded = self.hparams.probs_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        for tokens in decoded:
            for token in self.hparams.probs_decoder.tokenizer.special_tokens:
                if token == self.hparams.probs_decoder.tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        return info, decoded

    def on_validation_epoch_start(self):
        """
        Called at the start of each validation epoch.
        """
        if self.trainer.sanity_checking:
            self.file_save_dir = os.path.join(self.hparams.save_dir, "dev", "sanity_check")
        else:
            self.file_save_dir = os.path.join(self.hparams.save_dir, "dev", f"epoch_{self.current_epoch}")
        os.makedirs(self.file_save_dir, exist_ok=True)

        self.output_file = os.path.join(self.file_save_dir, f'output-hypothesis.txt')

        if self.trainer.is_global_zero:
            # 清空文件
            self.write_to_file("", "w")

        # 同步所有进程
        torch.distributed.barrier()

    def on_test_epoch_start(self):
        """
        Called at the start of each test epoch.
        """
        self.file_save_dir = os.path.join(self.hparams.save_dir, "test", f"test_after_epoch_{self.current_epoch}")
        os.makedirs(self.file_save_dir, exist_ok=True)

        self.output_file = os.path.join(self.file_save_dir, f'output-hypothesis.txt')

        if self.trainer.is_global_zero:
            # 清空文件
            self.write_to_file("", "w")

        # 同步所有进程
        torch.distributed.barrier()

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        torch.distributed.barrier()

        if self.trainer.is_global_zero:
            with open(self.lock_file, 'w') as f:
                # 获取文件锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    with open(self.output_file, "r") as output_file:
                        total_names = [line.split()[0] for line in output_file if line.strip()]
                        assert len(total_names) == len(set(total_names))
                finally:
                    # 释放文件锁
                    fcntl.flock(f, fcntl.LOCK_UN)

            try:
                # Call evaluate function to compute WER
                wer = self.hparams.evaluator.evaluate(
                    save_dir=self.file_save_dir,
                    hyp_file=self.output_file,
                    lock_file=self.lock_file,
                    mode="dev"
                )
            except Exception as e:
                # Handle exceptions and log error information
                print(f"ERROR: Exception occurred at the end of validation epoch: {e},",
                      f"please check detailed error message.")
                wer = '100.0'
            finally:
                # Process WER logging, ensuring even string values are logged correctly
                if isinstance(wer, str):
                    wer = float(re.findall("\d+\.?\d*", wer)[0])
                # Log DEV_WER metric
                self.log('Val/Word-Error-Rate', wer,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)
                # Print different messages based on whether it's a sanity check
                if self.trainer.sanity_checking:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Sanity Check, DEV_WER: {wer}%")
                else:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {self.current_epoch}, DEV_WER: {wer}%")

    def on_test_epoch_end(self):
        """
        Called at the end of each test epoch.
        """
        torch.distributed.barrier()

        if self.trainer.is_global_zero:
            with open(self.lock_file, 'w') as f:
                # 获取文件锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    with open(self.output_file, "r") as output_file:
                        total_names = [line.split()[0] for line in output_file if line.strip()]
                        assert len(total_names) == len(set(total_names))
                finally:
                    # 释放文件锁
                    fcntl.flock(f, fcntl.LOCK_UN)

            try:
                # Call evaluate function to compute WER
                wer = self.hparams.evaluator.evaluate(
                    save_dir=self.file_save_dir,
                    hyp_file=self.output_file,
                    lock_file=self.lock_file,
                    mode="test"
                )
            except Exception as e:
                # Handle exceptions and log error information
                print(f"ERROR: Exception occurred at the end of the test epoch: {e},",
                      f"please check the detailed error message.")
                wer = '100.0'
            finally:
                # Process WER logging, ensuring even string values are logged correctly
                if isinstance(wer, str):
                    wer = float(re.findall("\d+\.?\d*", wer)[0])
                # Log TEST_WER metric
                self.log('Test/Word-Error-Rate', wer,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)
                # Print messages
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Test after epoch {self.current_epoch - 1},",
                      f"TEST_WER: {wer}%")

    def write_to_file(self, data, open_mode='a'):
        with open(self.lock_file, 'w') as f:
            # 获取文件锁
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # 写入数据
                with open(self.output_file, open_mode) as output_file:
                    output_file.write(data)
            finally:
                # 释放文件锁
                fcntl.flock(f, fcntl.LOCK_UN)

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
            milestones = self.hparams.milestones
            gamma = self.hparams.gamma
            last_epoch = getattr(self.hparams, 'last_epoch', -1)  # Default value
        except AttributeError as e:
            # Raise an error if required hyperparameters are missing
            raise ValueError(f"Missing required hparam: {e}")

        # Initialize the Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
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
