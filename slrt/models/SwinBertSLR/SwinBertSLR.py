from typing import Tuple, Any

import torch
from torch import nn
from transformers import SwinModel

from slrt.models.BaseModel.SLRTBaseModel import SLRTBaseModel
from slrt.models.CorrNet.modules import Identity, TemporalConv, NormLinear, BiLSTMLayer, SeqKD


class SwinBertSLR(SLRTBaseModel):
    """
    A custom neural network architecture designed for Sign Language Recognition tasks.

    This class implements a model that combines Swin Transformer for spatial feature extraction,
    1D convolutional layers for temporal feature extraction, and BiLSTM layers for sequence modeling.
    It also includes an attention mechanism to align visual and text features, and a KL divergence loss for better alignment.

    Attributes:
        name (str): The name of the model.
        visual_feature_extractor (nn.Module): Swin Transformer for spatial feature extraction.
        text_feature_extractor (nn.Module): Bert for text feature extraction.
        temporal_feature_extractor (nn.Module): 1D convolutional layers for temporal feature extraction.
        sequence_modeler (nn.Module): BiLSTM layers for sequence modeling.
        attention (nn.Module): Attention mechanism for aligning visual and text features.
        classifier (nn.Module): Classifier for final prediction.
        ctc_loss (nn.Module): CTC loss function.
        dist_loss (nn.Module): Distillation loss function.
        kl_loss (nn.Module): KL divergence loss function.
    """

    def __init__(self, **kwargs):
        """
        Initializes the SwinBertSLR model with the given hyperparameters.

        Args:
            **kwargs: Hyperparameters for the model, including:
                - swin_pretrained (bool): Whether to use a pretrained Swin Transformer model.
                - bert_pretrained (bool): Whether to use a pretrained Bert model.
                - hidden_size (int): Size of the hidden layers.
                - conv_type (str): Type of convolution to use in the 1D convolutional layers.
                - use_bn (bool): Whether to use batch normalization in the 1D convolutional layers.
                - share_classifier (bool): Whether to share the classifier between 1D and 2D convolutional layers.
                - gloss_dict (dict): Dictionary mapping glosses to indices.
                - loss_weights (list): Weights for different loss components.
        """
        super().__init__(**kwargs)
        self.name = "SwinBertSLR"

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
        # Initialize Swin Transformer
        self.visual_feature_extractor = SwinModel.from_pretrained(
            self.hparams.swin_pretrained_model) if self.hparams.swin_pretrained else SwinModel()
        self.visual_feature_extractor.head = Identity()  # Replace the head with an Identity layer
        self.conv1 = nn.Conv1d(in_channels=49, out_channels=20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)

        # Initialize Bert
        # self.text_feature_extractor = BertModel.from_pretrained(
        #     self.hparams.bert_pretrained_model) if self.hparams.bert_pretrained else BertModel()

        # Initialize 1D Convolutional Layers
        self.temporal_feature_extractor = TemporalConv(
            input_size=5 * 768,  # Swin Transformer output size
            hidden_size=self.hparams.hidden_size,
            conv_type=self.hparams.conv_type,
            use_bn=self.hparams.use_bn,
            num_classes=self.hparams.probs_decoder.tokenizer.vocab_size
        )
        self.temporal_feature_extractor.fc = NormLinear(
            self.hparams.hidden_size,
            self.hparams.probs_decoder.tokenizer.vocab_size
        )  # Set the classifier

        # Initialize BiLSTM Layers
        self.sequence_modeler = BiLSTMLayer(
            rnn_type='LSTM',
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=2,
            bidirectional=True
        )

        # Initialize Attention Mechanism
        # self.attention = Attention(query_dim=self.hparams.hidden_size, key_dim=768, value_dim=768)

        # Initialize Classifier
        if self.hparams.share_classifier:
            self.classifier = self.temporal_feature_extractor.fc  # Use the same classifier as the 1D convolutional layer
        else:
            self.classifier = NormLinear(self.hparams.hidden_size, self.hparams.num_classes)  # Create a new classifier

    def forward(self, x: torch.Tensor, x_lgt: torch.Tensor, text: torch.Tensor, text_lgt: torch.Tensor) -> Tuple[
        Any, Any, Any]:  # , Any, Any]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width).
            x_lgt (torch.Tensor): Lengths of the sequences in the batch.
            text (torch.Tensor): Input text tensor of shape (batch_size, text_sequence_length).
            text_lgt (torch.Tensor): Lengths of the text sequences in the batch.

        Returns:
            Tuple[Any, Any, Any, Any]: Output logits from the 1D convolutional layer, output logits from the classifier, feature lengths, and aligned visual features.
        """
        batch_size, sequence_length, channels, height, width = x.shape
        # Reshape the input tensor
        reshaped_inputs = x.permute(0, 2, 1, 3, 4)

        # Pass through Swin Transformer
        visual_features = []
        for i in range(sequence_length):
            frame = reshaped_inputs[:, :, i, :, :]
            feature = self.visual_feature_extractor(frame).last_hidden_state
            visual_features.append(feature)
        visual_features = torch.stack(visual_features, dim=1)  # (batch_size, sequence_length,patch feature_dim)
        visual_features = visual_features.view(-1, 49, 768)
        visual_features = self.conv1(visual_features)
        visual_features = self.conv2(visual_features)
        visual_features = visual_features.view(batch_size, sequence_length, 5, 768)
        visual_features = visual_features.view(batch_size, sequence_length, -1)
        visual_features = visual_features.permute(0, 2, 1)

        # Pass through 1D convolutional layers
        conv1d_output = self.temporal_feature_extractor(visual_features, x_lgt)
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

        # Pass through Bert
        # text_features = self.text_feature_extractor(text, attention_mask=(text != 0)).last_hidden_state

        # Apply attention mechanism to align visual and text features
        # aligned_visual_features = None  # self.attention(queries=predictions, keys=text_features, values=text_features)

        return conv1d_logits, output_logits, feature_lengths  # , aligned_visual_features, text_features

    def _define_loss_function(self):
        """
        Defines the loss functions used by the model.
        """
        self.ctc_loss = nn.CTCLoss(reduction='none', zero_infinity=False)  # Define CTC loss function
        self.dist_loss = SeqKD(T=8)  # Define distillation loss function
        # self.kl_loss = KLDivLoss()  # Define KL divergence loss function

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
        # conv1d_hat, y_hat, y_hat_lgt, aligned_visual_features, text_features = self(x, x_lgt, y, y_lgt)
        conv1d_hat, y_hat, y_hat_lgt = self(x, x_lgt, y, y_lgt)

        # Calculate CTC loss
        ctc_loss_value = self.hparams.loss_weights[0] * self.ctc_loss(conv1d_hat.log_softmax(-1), y, y_hat_lgt,
                                                                      y_lgt).mean()
        ctc_loss_value += self.hparams.loss_weights[1] * self.ctc_loss(y_hat.log_softmax(-1), y, y_hat_lgt,
                                                                       y_lgt).mean()

        # Calculate distillation loss
        dist_loss_value = self.hparams.loss_weights[2] * self.dist_loss(conv1d_hat, y_hat.detach(), use_blank=False)

        # Calculate KL divergence loss
        # kl_loss_value = self.hparams.loss_weights[3] * self.kl_loss(aligned_visual_features, text_features)

        # Total loss
        loss = ctc_loss_value + dist_loss_value  # + kl_loss_value

        # Check for NaN values
        if torch.isnan(loss):
            print('\nWARNING: Detected NaN in loss.')

        return loss, conv1d_hat, y_hat_lgt, info

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

