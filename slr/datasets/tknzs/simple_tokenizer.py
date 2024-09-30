from typing import List, Union, Dict, Tuple

import torch


class SimpleTokenizer:
    """
    A simple tokenizer that converts glosses into numerical tokens.

    Attributes:
        vocab (dict): Vocabulary mapping words to integers.
        vocab_size (int): Size of the vocabulary.
        ids_to_vocab (dict): Inverse mapping from integers to words.
    """

    def __init__(
            self,
            vocab_file: str = None,
            pad_token: str = "<PAD>",
            unk_token: str = "<UNK>",
            sos_token: str = "<SOS>",
            eos_token: str = "<EOS>"
    ):
        """
        Initialize the tokenizer with an optional predefined vocabulary and special tokens.

        Args:
            vocab_file (str, optional): Path to the vocabulary file.
            pad_token (str): Padding token.
            unk_token (str): Unknown token.
            sos_token (str): Start of sequence token.
            eos_token (str): End of sequence token.
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        if vocab_file:
            self.vocab, self.ids_to_vocab = self.load_vocab(vocab_file)
        else:
            self.vocab = {}
            self.ids_to_vocab = {}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Loads the vocabulary from a file where each line contains one token.

        Args:
            vocab_file (str): Path to the vocabulary file.

        Returns:
            tuple(dict, dict): Vocabulary mapping words to integers and inverse mapping.
        """
        vocab = {}
        ids_to_vocab = {}

        # Add special tokens at the beginning with specific IDs
        special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            vocab[token] = i
            ids_to_vocab[i] = token

        # Load tokens from the file
        with open(vocab_file, 'r', encoding='utf-8') as f:
            index = len(special_tokens)  # Start indexing after special tokens
            for line in f:
                word = line.strip()
                if word:  # Ignore empty lines
                    if word in special_tokens:
                        continue
                    if word in vocab:
                        print(f"Warning: Duplicate token '{word}' found. Skipping.")
                        continue
                    vocab[word] = index
                    ids_to_vocab[index] = word
                    index += 1

        return vocab, ids_to_vocab

    def encode(
            self,
            text: Union[str, List[str]],
            return_type: str = 'pt',
            add_special_tokens: bool = True
    ) -> Union[List[int], torch.Tensor]:
        """
        Encodes a text into a list of token indices. Optionally adds <SOS> and <EOS> tokens.

        Args:
            text (str or list): The text to encode (string or list of words).
            return_type (str, optional): The type of the returned tensor ('list', 'pt').
            add_special_tokens (bool, optional): Whether to add <SOS> and <EOS> tokens.

        Returns:
            list or torch.Tensor: List or tensor of token indices representing the encoded text.
        """
        if isinstance(text, str):
            tokens = [self.vocab[word] if word in self.vocab else self.vocab[self.unk_token] for word in
                      self.tokenize(text)]
        elif isinstance(text, list):
            tokens = [self.vocab[word] if word in self.vocab else self.vocab[self.unk_token] for word in text]
        else:
            raise ValueError("Unsupported text type. Expected 'str' or 'list'.")

        if add_special_tokens:
            tokens = [self.vocab[self.sos_token]] + tokens + [self.vocab[self.eos_token]]

        if return_type == 'list':
            return tokens
        elif return_type == 'pt':
            return torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")

    def decode(
            self,
            token_indices: Union[List[int], torch.Tensor]
    ) -> List[str]:
        """
        Decodes a list of token indices back into a list of tokens.

        Args:
            token_indices (list or torch.Tensor): List or tensor of token indices.

        Returns:
            List[str]: List of decoded tokens.
        """
        if isinstance(token_indices, torch.Tensor):
            token_indices = token_indices.tolist()

        decoded_tokens = [self.convert_ids_to_tokens(token_index) for token_index in token_indices]
        return decoded_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a single token or a list of tokens to their corresponding ids.

        Args:
            tokens (str or list): The tokens to convert.

        Returns:
            int or list: The corresponding ids.
        """
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.vocab[self.unk_token])
        else:
            return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Converts a single id or a list of ids to their corresponding tokens.

        Args:
            ids (int or list): The ids to convert.

        Returns:
            str or list: The corresponding tokens.
        """
        if isinstance(ids, int):
            return self.ids_to_vocab.get(ids, self.unk_token)
        else:
            return [self.ids_to_vocab.get(id, self.unk_token) for id in ids]

    def save_vocabulary(self, save_path: str):
        """
        Saves the vocabulary to a file.

        Args:
            save_path (str): Path to save the vocabulary file.
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            for word in self.vocab.keys():
                f.write(f"{word}\n")

    def tokenize(self, sentence: str) -> List[str]:
        """
        Tokenizes a given sentence into a list of tokens.

        Args:
            sentence (str): The input sentence to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        # 默认按空格分词
        tokens = [word for word in sentence.split(' ') if word]
        return tokens
