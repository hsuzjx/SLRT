from itertools import groupby
from typing import Any

import tensorflow as tf
import torch

tf.config.set_visible_devices([], 'GPU')


class TFCTCBeamSearchDecoder:
    """
    CTC Beam Search Decoder class for converting neural network outputs into text using beam search.

    Attributes:
        tokenizer (Any): Tokenizer object used for encoding and decoding text.
        ctc_decoder (ctcdecode.CTCBeamDecoder): The beam search decoder initialized with the vocabulary.
    """

    def __init__(self, tokenizer: Any, beam_width: int = 1, top_paths: int = 1):
        """
        Initializes the CTC Beam Search Decoder.

        Args:
            tokenizer (Any): Tokenizer object used for encoding and decoding text. It must have `vocab`, `convert_tokens_to_ids`, and `decode` methods.
            beam_width (int, optional): Width of the beam during the search. Defaults to 10.
            num_processes (int, optional): Number of processes used by the beam search decoder. Defaults to 10.
        """
        self.tokenizer = tokenizer

        self.beam_width = beam_width
        self.top_paths = top_paths

    def decode(self, network_output: torch.Tensor, sequence_lengths: torch.Tensor, batch_first: bool = True) -> list:
        """
        Decodes the output of a neural network using beam search.

        Args:
            network_output (torch.Tensor): Output of the neural network, probabilities after softmax.
                                            Shape is (B, T, N) if batch_first is True, otherwise (T, B, N).
                                            The tensor should be on CPU.
            sequence_lengths (torch.Tensor): Lengths of the sequences in the batch (B). The tensor should be on CPU.
            batch_first (bool, optional): If True, the input is expected to be in the format (B, T, N).
                                          Otherwise, it is expected to be in the format (T, B, N).
                                          Defaults to False.

        Returns:
            list: List of decoded texts.
        """
        # Ensure tensors are on CPU and permute if necessary
        network_output = network_output.cpu()
        sequence_lengths = sequence_lengths.cpu()

        # Convert to TensorFlow
        tf_inputs = tf.convert_to_tensor(network_output.numpy())
        tf_sequence_length = tf.convert_to_tensor(sequence_lengths.numpy(), dtype=tf.int32)

        # Permute if necessary
        if not batch_first:
            tf_inputs = tf.transpose(tf_inputs, perm=[1, 0, 2])

        # Perform beam search decoding
        return self._perform_beam_search(tf_inputs, tf_sequence_length)

    def _perform_beam_search(self, tf_inputs, tf_sequence_length) -> list:
        """
        Internal method to perform beam search decoding.

        Args:


        Returns:
            list: List of decoded texts.
        """

        # Decode the input tensor using the CTC decoder
        beam_results, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf_inputs,
            sequence_length=tf_sequence_length,
            beam_width=self.beam_width,
            top_paths=self.top_paths
        )

        best_beam_result = beam_results[0]

        tmp_gloss_sequences = [[] for i in range(len(tf_sequence_length))]
        for (value_idx, dense_idx) in enumerate(best_beam_result.indices):
            tmp_gloss_sequences[dense_idx[0]].append(
                best_beam_result.values[value_idx].numpy() + 1
            )

        # Process the decoding results
        decoded_texts = []
        for batch_index in range(len(tf_sequence_length)):
            # Select the top beam result and trim it based on the output length
            # Remove consecutive duplicate tokens
            best_result = torch.tensor([x[0] for x in groupby(tmp_gloss_sequences[batch_index])])
            # Convert the token IDs back to text
            decoded_texts.append(self.tokenizer.decode(best_result))

        return decoded_texts
