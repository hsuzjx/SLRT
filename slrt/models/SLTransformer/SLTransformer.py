from typing import Any

import torch
import transformers
from torch import nn, Tensor

from slrt.models.BaseModel.SLRBaseModel import SLRBaseModel


class SLTransformer(SLRBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "SLTransformer"

        # 定义网络
        self._init_network()

        # 定义解码器
        self._define_decoder()

        # 定义损失函数
        self._define_loss_function()

    def _init_network(
            self, **kwargs
    ):
        """
        """
        super().__init__()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hparams.tokenizer.model_name
        )

        self.tokenizer = transformers.BertTokenizer(vocab_file=self.hparams.tokenizer.vocab_file)
        self.model = transformers.BertModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.encoder = transformers.AutoModel.from_pretrained(
            self.hparams.encoder.model_name
        )
        self.decoder = transformers.AutoModelForCausalLM.from_pretrained(
            self.hparams.decoder.model_name
        )
        self.gloss_output_layer = nn.Linear(
            self.hparams.encoder.hidden_size, self.hparams.gloss_vocab_size
        )
        self.txt_embed = nn.Embedding(
            self.hparams.txt_vocab_size, self.hparams.txt_embed_size
        )
        self.sgn_embed = nn.Embedding(
            self.hparams.sgn_vocab_size, self.hparams.sgn_embed_size
        )

    # pylint: disable=arguments-differ
    def forward(
            self,
            sgn: Tensor,
            sgn_mask: Tensor,
            sgn_lengths: Tensor,
            txt_input: Tensor,
            txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )
        decoder_outputs, gloss_probabilities = self.decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            sgn_mask=sgn_mask,
            txt_input=txt_input,
            unroll_steps=self.hparams.max_txt_length,
            txt_mask=txt_mask,
        )
        gloss_probabilities = gloss_probabilities.view(
            gloss_probabilities.size(0) * gloss_probabilities.size(1),
            gloss_probabilities.size(2),
        )
        decoder_outputs = decoder_outputs.view(
            decoder_outputs.size(0) * decoder_outputs.size(1),
            decoder_outputs.size(2),
            decoder_outputs.size(3),
        )
        gloss_probabilities = gloss_probabilities.view(
            gloss_probabilities.size(0) // self.hparams.batch_size,
            self.hparams.batch_size,
            gloss_probabilities.size(1),
        )
        decoder_outputs = decoder_outputs.view(
            decoder_outputs.size(0) // self.hparams.batch_size,
            self.hparams.batch_size,
            decoder_outputs.size(1),
            decoder_outputs.size(2),
        )
        gloss_probabilities = gloss_probabilities.permute(0, 2, 1)

        return decoder_outputs, gloss_probabilities

    def encode(
            self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
            self,
            encoder_output: Tensor,
            encoder_hidden: Tensor,
            sgn_mask: Tensor,
            txt_input: Tensor,
            unroll_steps: int,
            decoder_hidden: Tensor = None,
            txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def step_forward(self, batch) -> Any:
        return self.step_forward_recognition(batch)

    def step_forward_recognition(self, batch) -> Any:
        """
        Forward pass through the model.
        """
        sgn, sgn_mask, sgn_lengths, txt_input, txt_mask = batch
        decoder_outputs, gloss_probabilities = self.forward(
            sgn=sgn,
            sgn_mask=sgn_mask,
            sgn_lengths=sgn_lengths,
            txt_input=txt_input,
            txt_mask=txt_mask,
        )
        return decoder_outputs, gloss_probabilities

    def _define_loss_function(self):
        pass

    def _define_decoder(self):
        pass

    def configure_optimizers(self):
        """
        """
        learning_rate = self.hparams.lr
        momentum = self.hparams.momentum
        weight_decay = self.hparams.weight_decay

        milestones = self.hparams.milestones
        gamma = self.hparams.gamma
        last_epoch = self.hparams.last_epoch

        # Define the optimizer
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=last_epoch
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
