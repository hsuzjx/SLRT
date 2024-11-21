from itertools import groupby
from typing import Any

import ctcdecode
import torch


class CTCBeamSearchDecoder:
    """
    CTC Beam Search Decoder class for converting neural network outputs into text using beam search.

    Attributes:
        tokenizer (Any): Tokenizer object used for encoding and decoding text.
        ctc_decoder (ctcdecode.CTCBeamDecoder): The beam search decoder initialized with the vocabulary.
    """

    def __init__(self, tokenizer: Any, beam_width: int = 10, num_processes: int = 10):
        """
        Initializes the CTC Beam Search Decoder.

        Args:
            tokenizer (Any): Tokenizer object used for encoding and decoding text. It must have `vocab`, `convert_tokens_to_ids`, and `decode` methods.
            beam_width (int, optional): Width of the beam during the search. Defaults to 10.
            num_processes (int, optional): Number of processes used by the beam search decoder. Defaults to 10.
        """
        self.tokenizer = tokenizer

        # Initialize the decoder with the vocabulary keys as labels, the specified beam width,
        # the ID of the blank token, and the number of processes for parallel execution.
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            labels=list(self.tokenizer.vocab.keys()),
            beam_width=beam_width,
            blank_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            num_processes=num_processes
        )

    def decode(self, network_output: torch.Tensor, sequence_lengths: torch.Tensor, batch_first: bool = False) -> list:
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
        if not batch_first:
            network_output = network_output.permute(1, 0, 2)

        # Perform beam search decoding
        return self._perform_beam_search(network_output, sequence_lengths)

    def _perform_beam_search(self, network_output: torch.Tensor, sequence_lengths: torch.Tensor) -> list:
        """
        Internal method to perform beam search decoding.

        Args:
            network_output (torch.Tensor): The output tensor from the neural network.
            sequence_lengths (torch.Tensor): The lengths of the sequences in the batch.

        Returns:
            list: List of decoded texts.
        """
        # Decode the input tensor using the CTC decoder
        beam_results, _, _, output_lengths = self.ctc_decoder.decode(network_output, sequence_lengths)

        # Process the decoding results
        decoded_texts = []
        for batch_index in range(len(sequence_lengths)):
            # Select the top beam result and trim it based on the output length
            best_result = beam_results[batch_index][0][:output_lengths[batch_index][0]]
            # Remove consecutive duplicate tokens
            best_result = torch.tensor([k for k, _ in groupby(best_result)])
            # Convert the token IDs back to text
            decoded_texts.append(self.tokenizer.decode(best_result))

        return decoded_texts
