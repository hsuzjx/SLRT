import argparse
import os.path

import cv2
import gradio as gr
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop

from slrt.constants import ModelClassDict
from slrt.datasets.transforms import ToTensor
from slrt.datasets.Datasets.VideoDatasets.utils import pad_video_sequence


class SingleSampleDataset(Dataset):
    """
    Custom Dataset class for handling a single video sample.

    This class is used to wrap a single video sample for processing and prediction.

    Attributes:
        video (torch.Tensor): The video tensor.
    """

    def __init__(self, video: torch.Tensor):
        """
        Initializes the dataset with the given video tensor.

        Args:
            video (torch.Tensor): The video tensor.
        """
        self.video = video

    def __len__(self) -> int:
        """
        Returns the length of the dataset, which is always 1 for a single sample.

        Returns:
            int: Length of the dataset.
        """
        return 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the video tensor at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            torch.Tensor: The video tensor.
        """
        return self.video

    def collate_fn(self, batch: tuple) -> tuple:
        """
        Collates a batch of video tensors.

        Pads the video sequences to ensure they have the same length.

        Args:
            batch (tuple): Tuple of video tensors.

        Returns:
            tuple: A tuple containing the padded video tensor, video lengths, and other placeholders.
        """
        video, video_length = pad_video_sequence(batch, batch_first=True, padding_value=0.0)
        video_length = torch.LongTensor(video_length)

        return video, None, video_length, None, None


class SLRWebUI:
    """
    Web UI for Sign Language Recognition.

    This class provides a Gradio interface for uploading videos and performing sign language recognition.

    Attributes:
        model (torch.nn.Module): The loaded model for sign language recognition.
        trainer (lightning.Trainer): The Lightning Trainer for running predictions.
        max_workers (int): Maximum number of workers for video processing.
    """

    def __init__(
            self,
            model_name: str,
            ckpt_file: str,
            accelerator: str = "cpu",
            devices: [str, list] = "auto",
            precision: str = "16-mixed",
            max_workers: int = 4
    ):
        """
        Initializes the SLRWebUI with the given parameters.

        Validates the model name and checkpoint file, and sets up the model and trainer.

        Args:
            model_name (str): Name of the model.
            ckpt_file (str): Path to the model checkpoint file.
            accelerator (str, optional): Accelerator to use. Defaults to "cpu".
            devices (str or list, optional): Devices to use. Defaults to "auto".
            precision (str, optional): Precision to use. Defaults to "16-mixed".
            max_workers (int, optional): Maximum number of workers for video processing. Defaults to 4.
        """
        if model_name not in ModelClassDict:
            print(f"Model name '{model_name}' is not supported.")
            exit(1)
        if not os.path.exists(ckpt_file):
            print(f"Checkpoint file '{ckpt_file}' does not exist.")
            exit(1)

        self.model = ModelClassDict[model_name].load_from_checkpoint(ckpt_file)
        self.trainer = L.Trainer(accelerator=accelerator, devices=devices, precision=precision, logger=False)
        self.max_workers = max_workers

    @staticmethod
    def read_and_transform_video(video_file: str, max_workers: int = 4) -> torch.Tensor:
        """
        Reads and transforms a video file into a tensor.

        Args:
            video_file (str): Path to the video file.
            max_workers (int, optional): Maximum number of workers for video processing. Defaults to 4.

        Returns:
            torch.Tensor: The transformed video tensor.
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise IOError(f"Failed to open video file: {video_file}")

            # Define the transformation for video frames
            transform = Compose([
                ToTensor(),
                Resize(256),
                CenterCrop(224),
                Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
            ])

            # frames = []
            # Use multi-threading to accelerate frame processing
            # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            #     futures = []
            #     frame_index = 0
            #     while True:
            #         ret, frame = cap.read()
            #         if not ret:
            #             break
            #         # Convert the frame to RGB format
            #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #         # Apply transformations to the frame
            #         futures.append((frame_index, executor.submit(transform, frame_rgb)))
            #         frame_index += 1
            #
            #     frames = [None] * len(futures)
            #     for index, future in concurrent.futures.as_completed(futures):
            #         frame_tensor = future.result()
            #         frames[index] = frame_tensor
            #
            # cap.release()
            # Check if all frames were processed
            # assert None not in frames

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transformations to the frame
                frames.append(frame_rgb)

            cap.release()
            # Stack all frame tensors into a single tensor
            video = transform(frames)
            return video

        except IOError as e:
            print(f"IOError: {e}")
            raise
        except cv2.error as e:
            print(f"OpenCV Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def predict(self, video: torch.Tensor) -> str:
        """
        Predicts the sign language sequence from the given video tensor.

        Args:
            video (torch.Tensor): The video tensor.

        Returns:
            str: The predicted sign language sequence.
        """
        # Wrap the single sample into a DataLoader
        ds = SingleSampleDataset(video)
        dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
        _, decoded = self.trainer.predict(self.model, dataloaders=dl)[0]

        return " ".join(decoded[0])

    def gr_fn(self, video_file: str) -> str:
        """
        Gradio function for processing the uploaded video and returning the prediction.

        Args:
            video_file (str): Path to the uploaded video file.

        Returns:
            str: The predicted sign language sequence.
        """
        # Read and transform the uploaded video
        video = self.read_and_transform_video(video_file, max_workers=self.max_workers)
        # Predict the sign language sequence
        hyp_text = self.predict(video)
        return hyp_text

    def launch(
            self,
            host: str = "localhost",
            port: int = 28081,
            browse: bool = False,
            share: bool = False
    ):
        """
        Launches the Gradio web interface for sign language recognition.

        Args:
            host (str, optional): Server address. Defaults to "localhost".
            port (int, optional): Server port. Defaults to 28081.
            browse (bool, optional): Open the service in the browser automatically. Defaults to False.
            share (bool, optional): Share the service link. Defaults to False.
        """
        # Initialize the Gradio interface
        iface = gr.Interface(
            fn=self.gr_fn,
            inputs=gr.Video(),
            outputs=gr.Text(label='Hypothesis'),
            title="Sign Language Recognition",
            description="",
            live=False,
            allow_flagging='never',
        )

        # Launch the Gradio interface
        iface.launch(
            server_name=host,
            server_port=port,
            inbrowser=browse,
            share=share
        )


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Sign Language Recognition Web UI")

    # Add command line arguments
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    parser.add_argument("--ckpt-file", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "gpu"], help="Accelerator to use")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision to use")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of workers for video processing")

    parser.add_argument("--host", type=str, default="localhost", help="Server address")
    parser.add_argument("--port", type=int, default=28081, help="Server port")
    parser.add_argument("--browse", action="store_true", help="Open the service in the browser automatically")
    parser.add_argument("--share", action="store_true", help="Share the service link")

    # Parse command line arguments
    args = parser.parse_args()

    # Argument validation
    if not args.model_name:
        parser.error("Model name is required")
    if not args.ckpt_file:
        parser.error("Checkpoint file path is required")
    if args.port < 1024 or args.port > 65535:
        parser.error("Port number must be between 1024 and 65535")

    # Initialize the web UI
    webui = SLRWebUI(
        model_name=args.model_name,
        ckpt_file=args.ckpt_file,
        accelerator=args.accelerator,
        devices=args.devices if args.devices == "auto" else [int(idx) for idx in args.devices.split(",")],
        precision=args.precision,
        max_workers=args.max_workers
    )

    # Start the web UI
    webui.launch(
        host=args.host,
        port=args.port,
        browse=args.browse,
        share=args.share
    )
