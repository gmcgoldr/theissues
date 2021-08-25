from typing import NamedTuple

import numpy as np
import torch

from .model import TransformerModel


class TrainContext(NamedTuple):
    model: torch.nn.Module
    # number of tokens in the vocabulary
    nvocab: int
    # run the model on sequences of this length
    seq_len: int
    device: str
    # the full text of tokens from which to create sequences
    tokens: torch.LongTensor
    optimizer: torch.optim.Optimizer
    # iterate this many batches in an epoch
    epoch_size: int
    # include this many sequences in a batch
    batch_size: int
    grad_clip: float


def build_batch(
    rng: np.random.BitGenerator, tokens: torch.LongTensor, seq_len: int, batch_size: int
) -> torch.LongTensor:
    """
    Build a batch of sequences of tokens from a larger sequence of tokens.

    Args:
        tokens: the full text from which to draw sequences
        seq_len: the number of tokens in a drawn sequence
        batch_size: aggregate this many sequences

    Returns:
        tensor of tokens with shape `(seq_len, batch_size)`
    """
    ntokens = tokens.size(0)
    offset = rng.choice(seq_len)
    ntokens -= offset
    nbatches = ntokens // seq_len
    batch_data = tokens[offset:]
    batch_data = (
        batch_data[: nbatches * seq_len].view((nbatches, seq_len)).transpose(0, 1)
    )
    indices = rng.choice(nbatches, batch_size)
    return batch_data[:, indices]


def train_epoch(ctx: TrainContext) -> float:
    """
    Train the model over one epoch using the provided configuration.

    Args:
        ctx: the training configuration

    Returns:
        the total loss over the epoch
    """
    ctx.model.train()

    rng = np.random.default_rng()
    src_mask = TransformerModel.build_subsequent_mask(ctx.seq_len).to(ctx.device)

    total_loss = 0.0

    # NOTE: an epoch is normally a full run over the data, but here it's used
    # to mean some fixed number of gradient steps which means reporting, saving
    # early stoping etc. is related to the amount of training, but exposure to
    # the full dataset.

    for _ in range(ctx.epoch_size):
        # sequence length needs +1 due to target being off by 1
        batch_data = build_batch(
            rng=rng,
            tokens=ctx.tokens,
            seq_len=ctx.seq_len + 1,
            batch_size=ctx.batch_size,
        )

        inputs = batch_data[:-1]
        targets = batch_data[1:]
        output = ctx.model(inputs, src_mask)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, ctx.nvocab), targets.view(-1)
        )
        total_loss += loss.item()

        ctx.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
        ctx.optimizer.step()

    return total_loss
