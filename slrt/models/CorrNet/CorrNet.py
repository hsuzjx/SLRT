from typing import Tuple, Any

import torch
from torch import nn

from slrt.models.BaseModel import SLRBaseModel
from slrt.models.CorrNet.modules import resnet18, Identity, TemporalConv, NormLinear, BiLSTMLayer, SeqKD


class CorrNet(SLRBaseModel):
    """
    A custom neural network architecture designed for Sign Language Recognition tasks.

    This class implements a model that combines 2D convolutional layers for spatial feature extraction,
    1D convolutional layers for temporal feature extraction, and BiLSTM layers for sequence modeling.
    It also includes methods for initializing the network components, defining the decoder, and setting up
    the loss functions.

    Attributes:
        name (str): The name of the model.
        spatial_feature_extractor (nn.Module): 2D convolutional layers for spatial feature extraction.
        temporal_feature_extractor (nn.Module): 1D convolutional layers for temporal feature extraction.
        sequence_modeler (nn.Module): BiLSTM layers for sequence modeling.
        classifier (nn.Module): Classifier for final prediction.
        ctc_loss (nn.Module): CTC loss function.
        dist_loss (nn.Module): Distillation loss function.
    """

    def __init__(self, **kwargs):
        """
        Initializes the CorrNet model with the given hyperparameters.

        Args:
            **kwargs: Hyperparameters for the model, including:
                - resnet_pretrained (bool): Whether to use a pretrained ResNet model.
                - pretrained_model_dir (str): Directory containing the pretrained model weights.
                - hidden_size (int): Size of the hidden layers.
                - conv_type (str): Type of convolution to use in the 1D convolutional layers.
                - use_bn (bool): Whether to use batch normalization in the 1D convolutional layers.
                - share_classifier (bool): Whether to share the classifier between 1D and 2D convolutional layers.
                - gloss_dict (dict): Dictionary mapping glosses to indices.
                - loss_weights (list): Weights for different loss components.
        """
        super().__init__(**kwargs)
        self.name = "CorrNet"

        # Initialize all network components
        self._init_network(**kwargs)

        # Define the loss functions
        self._define_loss_function()

    def _init_network(self, **kwargs):
        """
        Initializes all network components.

        Args:
            **kwargs: Hyperparameters for the model.
        """
        # Initialize 2D Convolutional Layers
        self.spatial_feature_extractor = resnet18(pretrained=self.hparams.resnet_pretrained,
                                                  model_dir=self.hparams.pretrained_model_dir)
        self.spatial_feature_extractor.fc = Identity()  # Replace the fully connected layer with an Identity layer

        # Initialize 1D Convolutional Layers
        self.temporal_feature_extractor = TemporalConv(
            input_size=512,
            hidden_size=self.hparams.hidden_size,
            conv_type=self.hparams.conv_type,
            use_bn=self.hparams.use_bn,
            num_classes=self.hparams.probs_decoder.tokenizer.vocab_size
        )
        self.temporal_feature_extractor.fc = NormLinear(self.hparams.hidden_size,
                                                        self.hparams.probs_decoder.tokenizer.vocab_size)  # Set the classifier

        # Initialize BiLSTM Layers
        self.sequence_modeler = BiLSTMLayer(
            rnn_type='LSTM',
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=2,
            bidirectional=True
        )

        # Initialize Classifier
        if self.hparams.share_classifier:
            self.classifier = self.temporal_feature_extractor.fc  # Use the same classifier as the 1D convolutional layer
        else:
            self.classifier = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)  # Create a new classifier

    def forward(self, x: torch.Tensor, x_lgt: torch.Tensor) -> Tuple[Any, Any, Any]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width).
            x_lgt (torch.Tensor): Lengths of the sequences in the batch.

        Returns:
            Tuple[Any, Any, Any]: Output logits from the 1D convolutional layer, output logits from the classifier, and feature lengths.
        """
        batch_size, sequence_length, channels, height, width = x.shape
        # Reshape the input tensor
        reshaped_inputs = x.permute(0, 2, 1, 3, 4)

        # Pass through 2D convolutional layers
        convolved = self.spatial_feature_extractor(reshaped_inputs)
        convolved = convolved.view(batch_size, sequence_length, -1).permute(0, 2, 1)

        # Pass through 1D convolutional layers
        conv1d_output = self.temporal_feature_extractor(convolved, x_lgt)
        # Get visual features
        visual_features = conv1d_output['visual_feat']
        # Get feature lengths
        feature_lengths = conv1d_output['feat_len']
        # Get 1D convolutional layer outputs
        conv1d_logits = conv1d_output['conv_logits']

        # Pass through BiLSTM layers
        lstm_output = self.sequence_modeler(visual_features, feature_lengths)

        # Get predictions
        predictions = lstm_output['predictions']
        # Pass through the classifier
        output_logits = self.classifier(predictions)

        return conv1d_logits, output_logits, feature_lengths

    def _define_loss_function(self):
        """
        Defines the loss functions used by the model.
        """
        self.ctc_loss = nn.CTCLoss(reduction='none', zero_infinity=False)  # Define CTC loss function
        self.dist_loss = SeqKD(T=8)  # Define distillation loss function

    def step_forward(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]) -> Tuple[
        torch.Tensor, Any, Any, Any]:
        """
        Performs a forward pass and computes the loss for a given batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]): A tuple containing the input data, input lengths, target data, target lengths, and additional information.

        Returns:
            Tuple[torch.Tensor, Any, Any, Any]: Loss value, softmax predictions, predicted lengths, and additional information.
        """
        x, y, x_lgt, y_lgt, info = batch
        conv1d_hat, y_hat, y_hat_lgt = self(x, x_lgt)

        if self.trainer.predicting:
            return torch.tensor([]), y_hat, y_hat_lgt, info

        loss = (
                self.hparams.loss_weights[0] * self.ctc_loss(conv1d_hat.log_softmax(-1), y, y_hat_lgt, y_lgt).mean() +
                self.hparams.loss_weights[1] * self.ctc_loss(y_hat.log_softmax(-1), y, y_hat_lgt, y_lgt).mean() +
                self.hparams.loss_weights[2] * self.dist_loss(conv1d_hat, y_hat.detach(), use_blank=False)
        )

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, y_hat, y_hat_lgt, info
