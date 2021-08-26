from typing import NamedTuple

import numpy as np
import sentencepiece as spm
import torch

from .model import TransformerModel


class TrainContext(NamedTuple):
    model: TransformerModel
    # number of tokens in the vocabulary
    nvocab: int
    # run the model on sequences of this length
    seq_len: int
    # make predictions with at least this many tokens as conditionals
    min_conditional: int
    device: str
    # the full text of tokens from which to create sequences
    tokens: torch.LongTensor
    optimizer: torch.optim.Optimizer
    # iterate this many batches in an epoch
    epoch_size: int
    # include this many sequences in a batch
    batch_size: int
    # batches can start at these indices in the `tokens`
    batch_indices: torch.LongTensor
    grad_clip: float


def build_batch_split_indices(
    tokens: torch.LongTensor, split_token_id: int
) -> torch.LongTensor:
    """
    Build an array with the indices in the `tokens` array of all tokens which
    match the `split_token_id`.
    """
    mask = tokens == split_token_id
    indices = torch.arange(tokens.size(0), dtype=torch.long)[mask]
    return indices


def build_batch_indices(
    rng: np.random.Generator,
    tokens: torch.LongTensor,
    split_idxs: torch.LongTensor,
    seq_len: int,
    batch_size: int,
) -> torch.LongTensor:
    """
    Build a tensor of token indices from which to build a batch of sequences.

    Args:
        rng: a random number generator used to select sequence locations
        tokens: the full tensor of tokens from which to build sequences
        split_idxs: indices in the `tokens` tensor where batches can start
        seq_len: build sequences with this many tokens
        batch_size: build a batch with this many sequences

    Returns:
        tensor of token indices in the `tokens` array from which to form a
        batch with shape `(seq_len, batch_size)`
    """
    indices = torch.arange(seq_len, dtype=torch.long)
    splits = rng.choice(split_idxs, batch_size)
    indices = indices[None, :] + splits[:, None]
    # if starting a sequence too close to the end of the data, will wrap around
    # and keep using the start as the next sequence
    indices = indices % tokens.size(0)
    indices = indices.transpose(0, 1)
    return indices


def train_epoch(ctx: TrainContext) -> float:
    """
    Train the model over one epoch using the provided configuration.

    Sequences have the form:

    `<src> src <bos> tok1 ...`

    Where `<src>` and `<bos>` are special tokens, `tok1 ...` is the
    sequence of vocabulary tokens to train from, and `src` is the sequence
    source token.

    `seq_len` includes the special tokens, so to predict a single vocabulary
    token (i.e. `tok1`) `seq_len` must be at least `4`. In this case the model
    will emit probabilites `P(tok1 | <src> src <bos>)`.

    Setting `min_conditional` will increase the number of context tokens in
    the conditional, e.g. predicting `tok2 ...` from `tok1 ...` for
    `min_conditional = 1`.

    Args:
        ctx: the training configuration

    Returns:
        the total loss over the epoch
    """
    if ctx.seq_len < 4:
        raise ValueError("`seq_len` must be >= 4")
    if 3 + ctx.min_conditional >= ctx.seq_len:
        raise ValueError("`min_conditional` must be < `seq_len - 3`")

    ctx.model.train()

    rng = np.random.default_rng()

    total_loss = 0.0

    # NOTE: an epoch is normally a full run over the data, but here it's used
    # to mean some fixed number of gradient steps which means reporting, saving
    # early stoping etc. is related to the amount of training, but exposure to
    # the full dataset.

    for _ in range(ctx.epoch_size):
        batch_indices = build_batch_indices(
            rng=rng,
            tokens=ctx.tokens,
            split_idxs=ctx.batch_indices,
            seq_len=ctx.seq_len,
            batch_size=ctx.batch_size,
        )

        batch_data = ctx.tokens[batch_indices]

        inputs = batch_data[:-1]
        targets = batch_data[1:]
        output = ctx.model(inputs)

        # the first loss term is predicting `tok1` from `<bos>`, and shifts
        # forward when `min_conditional > 0`.
        offset = 2 + ctx.min_conditional
        output = output[offset:]
        targets = targets[offset:]

        loss = torch.nn.functional.cross_entropy(
            # NOTE: reshape because `batch_data` isn't contiguous, setting
            # contiguous upstream has no appreciable effect on performance
            output.reshape(-1, ctx.nvocab),
            targets.reshape(-1),
        )
        total_loss += loss.item()

        ctx.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
        ctx.optimizer.step()

    return total_loss


class GeneratorContext(NamedTuple):
    model: torch.nn.Module
    tokenizer: spm.SentencePieceProcessor
    temperature: float
    max_tokens: int


def generate_seq(ctx: GeneratorContext, seed: str = None, source: str = None):
    ctx.model.eval()

    try:
        device = next(ctx.model.parameters()).device
    except StopIteration:
        device = "cuda" if torch.nn.cuda.cuda.is_available() else "cpu"

    src_id = ctx.tokenizer.piece_to_id("<src>")
    bos_id = ctx.tokenizer.bos_id()
    unk_id = ctx.tokenizer.unk_id()

    if unk_id in {src_id, bos_id}:
        raise RuntimeError("tokenizer must have <s> and <src> tokens")

    input = [
        src_id,
        ctx.tokenizer.piece_to_id(source) if source else unk_id,
        bos_id,
    ]
    if seed:
        input += ctx.tokenizer.encode(seed)

    # the tokens to show start after <bos>
    token_idxs = list(input[3:])

    input = torch.LongTensor(input).to(device)

    with torch.no_grad():  # no tracking history
        while input.size(0) < ctx.max_tokens:
            # add the (single) batch dimension at index 1
            output = ctx.model(input.unsqueeze(1))
            # select the prediction from the last token
            output = output[-1]
            # temperature can squash (expand) the logit values, and the relative
            # probs. is proportional to the differences, which are all smaller
            # (larger) so the things are more (less) evenly distributed
            output = output / ctx.temperature
            # run through softmax to get probs, but `multinomial` will also take
            # care of normalization so can just use `exp` here
            probs = torch.exp(output)

            # draw a single token index from the probs
            token_idx = torch.multinomial(probs.cpu(), 1)[0].item()

            # add the word to the input
            input = torch.cat([input, torch.LongTensor([token_idx]).to(device)])

            # when "end of string" is emitted, break early
            if token_idx == ctx.tokenizer.eos_id():
                break

            token_idxs.append(token_idx)

    if token_idxs:
        return ctx.tokenizer.decode(token_idxs)
    else:
        return ""
