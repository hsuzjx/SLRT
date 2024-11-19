import unittest
from unittest.mock import patch, mock_open

import torch

from slr.datasets.text_tokenizers.simple_tokenizer import SimpleTokenizer


class TestSimpleTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = SimpleTokenizer()

    def test_load_vocab(self):
        # 测试加载一个简单的词汇表
        vocab_content = "word1\nword2\nword3\n"
        with patch('builtins.open', mock_open(read_data=vocab_content)) as mock_file:
            mock_file.return_value.__enter__.return_value = mock_file
            vocab, ids_to_vocab = self.tokenizer.load_vocab('fake_vocab.txt')
            self.assertEqual(vocab['word1'], 1)
            self.assertEqual(ids_to_vocab[2], 'word2')
            self.assertEqual(len(vocab), 7)  # 3 words + 4 special tokens
            self.assertIn('<PAD>', vocab)
            self.assertIn('<UNK>', ids_to_vocab)

    def test_encode_decode(self):
        # 测试编码和解码
        self.tokenizer.vocab = {
            'hello': 1,
            'world': 2,
            '<PAD>': 0,
            '<UNK>': 3,
            '<SOS>': 4,
            '<EOS>': 5
        }
        self.tokenizer.ids_to_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

        text = 'hello world'
        encoded = self.tokenizer.encode(text)
        expected = torch.tensor([1, 2], dtype=torch.long)
        self.assertTrue(torch.equal(encoded, expected), "Encoded tensor does not match expected output.")

        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, ['hello', 'world'], "Decoded tokens do not match expected output.")

    def test_save_vocabulary(self):
        # 测试保存词汇表
        save_path = 'test_vocab.txt'
        self.tokenizer.vocab = {
            'hello': 1,
            'world': 2,
        }
        self.tokenizer.ids_to_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value = mock_file
            self.tokenizer.save_vocabulary(save_path)
            mock_file().write.assert_called_with('hello\nworld\n')

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
