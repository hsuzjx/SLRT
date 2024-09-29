from torchaudio.models.decoder import ctc_decoder


class CTCBeamSearchDecode(object):
    """
    """

    def __init__(self, tokenizer, nbest=10, beam_size=100, beam_size_token=None, beam_threshold=50):

        self.tokenizer = tokenizer

        self.decoder = ctc_decoder(
            tokens=self.tokenizer.vocab.keys(),
            nbest=nbest,
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            log_add=False,
            blank_token=self.tokenizer.pad_token,
            unk_word=self.tokenizer.unk_token,
        )

    def decode(self, nn_output, vid_lgt, batch_first=True):
        """
        """
        # 如果不是批量优先，调整输入维度顺序
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)

        return self.decoder.decode(nn_output, vid_lgt)
