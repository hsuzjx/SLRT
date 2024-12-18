import os
import re
from datetime import datetime
from typing import Any, Union, Sequence

import lightning as L
import torch
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


class SLRTBaseModel(L.LightningModule):
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
        self.hypothesis_save_dir = os.path.join(os.path.abspath(self.hparams.save_dir), 'hypothesis')
        self.checkpoint_save_dir = os.path.join(os.path.abspath(self.hparams.save_dir), 'checkpoints')
        self.file_save_dir = None
        self.output_file = None
        self.rank_output_file = None

        self.recognition_decoder = self.hparams.probs_decoder['recognition'] \
            if 'recognition' in self.hparams.probs_decoder.keys() else None
        self.translation_decoder = self.hparams.probs_decoder['translation'] \
            if 'translation' in self.hparams.probs_decoder.keys() else None

        self.recognition_tokenizer = self.recognition_decoder.tokenizer \
            if self.recognition_decoder else None
        self.translation_tokenizer = self.translation_decoder.tokenizer \
            if self.translation_decoder else None

        self.recognition_evaluator = self.hparams.evaluator['recognition'] \
            if 'recognition' in self.hparams.evaluator.keys() else None
        self.translation_evaluator = self.hparams.evaluator['translation'] \
            if 'translation' in self.hparams.evaluator.keys() else None

        # Register hook for handling NaN gradients
        self.register_full_backward_hook(self.handle_nan_gradients)

    def set_decoder(self, decoder, task='recognition'):
        if task == 'recognition':
            self.recognition_decoder = decoder
            self.recognition_tokenizer = decoder.tokenizer
            print('Recognition decoder and tokenizer is set')
        elif task == 'translation':
            self.translation_decoder = decoder
            self.translation_tokenizer = decoder.tokenizer
            print('Translation decoder and tokenizer is set')
        else:
            print('Task not supported, the supported task is in [\'recognition\', \'translation\']')

    def set_evaluator(self, evaluator, task='recognition'):
        if task == 'recognition':
            self.recognition_evaluator = evaluator
            print('Recognition evaluator is set')
        elif task == 'translation':
            self.translation_evaluator = evaluator
            print('Translation evaluator is set')
        else:
            print('Task not supported, the supported task is in [\'recognition\', \'translation\']')

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
        decoded = self.recognition_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        # Remove special tokens from the decoded predictions
        for tokens in decoded:
            for token in self.recognition_tokenizer.special_tokens:
                if token == self.recognition_tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Write the decoded predictions to the output file
        for i in range(len(info)):
            self.rank_output_file.write(f"{info[i]} {' '.join(decoded[i])}\n")

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
        decoded = self.recognition_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        # Remove special tokens from the decoded predictions
        for tokens in decoded:
            for token in self.recognition_tokenizer.special_tokens:
                if token == self.recognition_tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        # Write the decoded predictions to the output file
        for i in range(len(info)):
            self.rank_output_file.write(f"{info[i]} {' '.join(decoded[i])}\n")

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
        decoded = self.glosses_decoder.decode(y_hat.softmax(-1).cpu(), y_hat_lgt.cpu())

        for tokens in decoded:
            for token in self.recognition_tokenizer.special_tokens:
                if token == self.recognition_tokenizer.unk_token:
                    continue
                while token in tokens:
                    tokens.remove(token)

        return info, decoded

    def on_validation_epoch_start(self):
        """
        Called at the start of each validation epoch.
        """
        if self.trainer.sanity_checking:
            self.file_save_dir = os.path.join(self.hypothesis_save_dir, "dev", "sanity_check")
        else:
            self.file_save_dir = os.path.join(self.hypothesis_save_dir, "dev", f"epoch_{self.current_epoch}")
        os.makedirs(self.file_save_dir, exist_ok=True)

        self.output_file = os.path.join(self.file_save_dir, f'output-hypothesis.txt')
        self.rank_output_file = open(
            os.path.join(self.file_save_dir, f'output-hypothesis-rank{self.trainer.global_rank}.txt'), "w")

    def on_test_epoch_start(self):
        """
        Called at the start of each test epoch.
        """
        self.file_save_dir = os.path.join(self.hypothesis_save_dir, "test",
                                          f"test_best_model_after_epoch_{self.current_epoch}")
        os.makedirs(self.file_save_dir, exist_ok=True)

        self.output_file = os.path.join(self.file_save_dir, f'output-hypothesis.txt')
        self.rank_output_file = open(
            os.path.join(self.file_save_dir, f'output-hypothesis-rank{self.trainer.global_rank}.txt'), "w")

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        self.rank_output_file.close()
        torch.distributed.barrier()

        wer = torch.tensor([100.0], device=self.device)

        if self.trainer.is_global_zero:
            all_lines = dict()
            for rank in range(self.trainer.world_size):
                rank_output_file = os.path.join(self.file_save_dir, f'output-hypothesis-rank{rank}.txt')
                with open(rank_output_file, 'r') as f:
                    for line in f.readlines():
                        all_lines[line.split(' ')[0]] = line

            with open(self.output_file, 'w') as f:
                f.writelines(list(all_lines.values()))

            try:
                # Call evaluate function to compute WER
                wer = self.recognition_evaluator.evaluate(
                    save_dir=self.file_save_dir,
                    hyp_file=self.output_file,
                    mode="dev"
                )
            except Exception as e:
                # Handle exceptions and log error information
                print(f"ERROR: Exception occurred at the end of validation epoch: {e},",
                      f"please check detailed error message.")
            finally:
                # Process WER logging, ensuring even string values are logged correctly
                if isinstance(wer, str):
                    wer = torch.tensor(float(re.findall("\d+\.?\d*", wer)[0]), device=self.device)
                if isinstance(wer, float):
                    wer = torch.tensor(wer, device=self.device)
                # Print different messages based on whether it's a sanity check
                if self.trainer.sanity_checking:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                          f"Sanity Check,",
                          f"DEV_WER: {wer.item()}%")
                else:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                          f"Epoch {self.current_epoch},",
                          f"DEV_WER: {wer.item()}%")

        torch.distributed.barrier()
        wer = self.all_gather(wer)[0]

        # Log DEV_WER metric
        self.log('Val/Word-Error-Rate', wer,
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=False, rank_zero_only=True)

    def on_test_epoch_end(self):
        """
        Called at the end of each test epoch.
        """
        self.rank_output_file.close()
        torch.distributed.barrier()

        wer = torch.tensor([100.0], device=self.device)

        if self.trainer.is_global_zero:
            all_lines = dict()
            for rank in range(self.trainer.world_size):
                rank_output_file = os.path.join(self.file_save_dir, f'output-hypothesis-rank{rank}.txt')
                with open(rank_output_file, 'r') as f:
                    for line in f.readlines():
                        all_lines[line.split(' ')[0]] = line

            with open(self.output_file, 'w') as f:
                f.writelines(list(all_lines.values()))

            try:
                # Call evaluate function to compute WER
                wer = self.recognition_evaluator.evaluate(
                    save_dir=self.file_save_dir,
                    hyp_file=self.output_file,
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
                    wer = torch.tensor(float(re.findall("\d+\.?\d*", wer)[0]), device=self.device)
                if isinstance(wer, float):
                    wer = torch.tensor(wer, device=self.device)
                # Print messages
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      f"Test best model after epoch {self.current_epoch - 1},",
                      f"TEST_WER: {wer.item()}%")

            torch.distributed.barrier()
            wer = self.all_gather(wer)[0]

            # Log TEST_WER metric
            self.log('Test/Word-Error-Rate', wer,
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=False, rank_zero_only=True)

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
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """

        :return:
        """
        early_stop = EarlyStopping(
            monitor='Val/Loss',
            patience=self.hparams.get('patience_early_stop', 40),
            verbose=True,
            mode='min',
            check_finite=False
        )
        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_save_dir,
            filename='top_{epoch}',
            monitor='Val/Word-Error-Rate',
            mode='min',
            save_last=self.hparams.save_last,
            save_top_k=self.hparams.save_top_k,
            verbose=True
        )
        return [early_stop, checkpoint]

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
