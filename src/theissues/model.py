"""
Transformer language model.

See: https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
"""

import math

import torch


class PositionalEncoding(torch.nn.Module):
    """
    Model which adds position encoding to an embedding space.

    The model has no learnable parameters.
    """

    def __init__(self, ndims: int, max_seq_len: int):
        """
        Args:
            ndims: encoder dimensions
            max_seq_len: maximum number tokens in a sequence
        """
        super().__init__()

        basis = torch.zeros(max_seq_len, ndims)
        # array with token index at each position
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        # prepare to broadcast accross the embedding dimension
        position = position[:, None]
        # Later dimensions get progressively longer trig. period, such that the
        # first dimension oscillates between 1 and -1 at each token, while the
        # last dimensions increase or decrease steadily.
        period = -math.log(2 * max_seq_len) / ndims
        period = torch.exp(torch.arange(0, ndims, 2).float() * period)
        # build the sin / cos basis
        basis[:, 0::2] = torch.sin(position * period * math.pi * 0.5)
        if ndims % 2 != 0:
            basis[:, 1::2] = torch.cos(position * period[:-1] * math.pi * 0.5)
        else:
            basis[:, 1::2] = torch.cos(position * period * math.pi * 0.5)
        # add batch dimension
        basis = basis[:, None, :]
        self.register_buffer("basis", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.basis[: x.size(0)]


class TransformerModel(torch.nn.Module):
    """
    Model maps a sequence of tokens to a sequence of proabilities over the
    vocabulary.

    Uses an embedding layer, a transformer and a decoder.
    """

    def __init__(
        self,
        nvocab: int,
        seq_len: int,
        ndims_embed: int,
        ndims_trans: int,
        nheads: int,
        nlayers: int,
        dropout: float,
    ):
        """
        Args:
            nvocab: number of vocabulary entries
            seq_len: number of tokens in a sequence
            ndims_embed: number of hidden dimensions in the embedding (and all
                token respresentations)
            ndims_trans: number of hidden dimensions in the transformer
            nheads: number of attention heads
            nlayers: number of transfomer layers
        """
        super().__init__()

        self.ndims_embed = ndims_embed

        self.positional_encoder = PositionalEncoding(
            ndims=ndims_embed, max_seq_len=seq_len
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=ndims_embed,
                nhead=nheads,
                dim_feedforward=ndims_trans,
                dropout=dropout,
            ),
            num_layers=nlayers,
        )
        self.embeddings = torch.nn.Embedding(nvocab, ndims_embed)
        self.decoder = torch.nn.Linear(ndims_embed, nvocab)
        self.dropout = torch.nn.Dropout(dropout)

        # store the attention mask with the model
        self.register_buffer(
            name="attention_mask", tensor=self.build_subsequent_mask(seq_len)
        )

        self.init_weights()

    @staticmethod
    def build_subsequent_mask(n: int) -> torch.FloatTensor:
        """
        Build an array with upper trinagular elements set to `-inf` and the
        rest along with diangonal set to `0`.

        With `n` set to a sequence length, this creates a mask which can be
        used to progressively allow each token in the sequence to propagate
        to the output.

        The entries add a weight to the attention of token `i` to `j`. The
        upper (past diagonal) entries have `-inf` attention weight for token `i`
        attending token `j > i`. Which is what is desired as token `i` will
        emit a prediction for token `j = i + 1`. In BERT-style training, only
        `i + 1` needs be masked, but to build an auto-regressive model capable
        of generating text, causality must be respected.
        """
        return torch.triu(torch.ones(n, n) * -math.inf, diagonal=1)

    def init_weights(self, scale: float = 0.1):
        torch.nn.init.uniform_(self.embeddings.weight, -scale, scale)
        torch.nn.init.zeros_(self.decoder.weight)
        torch.nn.init.uniform_(self.decoder.weight, -scale, scale)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # convert the token indieces to the embedding space, with positional
        # encodings and dropout applied
        x = self.embeddings(x) * math.sqrt(self.ndims_embed)
        x = self.positional_encoder(x)
        x = self.dropout(x)

        # shorten the mask if run on smaller sequences
        n = x.size(0)
        attention_mask = self.attention_mask[:n, :n]

        # run through the transformer into the output feature space
        x = self.transformer_encoder(x, attention_mask)
        # decode the features into token weights
        x = self.decoder(x)

        # NOTE: `x` is now appropriate for `CrossEntropyLoss`, but to get the
        # token probabilities, `Softmax` must be applied.

        return x
