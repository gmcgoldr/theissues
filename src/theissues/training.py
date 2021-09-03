import math
from typing import List, NamedTuple

import numpy as np
import tokenizers as tk
import torch

from .model import TransformerModel
from .utils import SpecialTokens


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
    rng: np.random.Generator


def build_token_splits(
    tokens: torch.LongTensor, split_token_id: int
) -> torch.LongTensor:
    """
    Build an array with the indices in the `tokens` array of all tokens which
    match the `split_token_id`.
    """
    mask = tokens == split_token_id
    indices = torch.arange(tokens.size(0), dtype=torch.long)[mask]
    return indices


def build_sequence_gather_indices(
    num_tokens: int,
    splits: torch.LongTensor,
    seq_len: int,
) -> torch.LongTensor:
    """
    Build a tensor of indices in a token tensor which can be used to gather
    a tensor of sequences with shape `(seq_len, num_seqs)` where `num_seqs`
    is the length of `splits`.

    Each sequence starts at the corresponding `split` index and runs for
    `seq_len` tokens. Sequences can run over one-another, and they loop around
    at the end of the `tokens` tensor.

    For example, given a tensor of 3 tokens, `seq_splits = [0, 1]` and
    `seq_len = 3`, the resulting tensor would be the transpose of:
    `[[0, 1, 2], [1, 2, 0]]`.

    Args:
        num_tokens: index in a tensor of this many tokens
        split_idxs: indices in the `tokens` tensor where each sequence starts
        seq_len: build sequences with this many tokens

    Returns:
        tensor of indices in the `tokens` tensor
    """
    splits = torch.as_tensor(splits, dtype=torch.long)
    indices = torch.arange(seq_len, dtype=torch.long)
    indices = indices[None, :] + splits[:, None]
    # if starting a sequence too close to the end of the data, will wrap around
    # and keep using the start as the next sequence
    indices = indices % num_tokens
    indices = indices.transpose(0, 1)
    return indices


def build_sequence_mask_after(
    sequences: torch.LongTensor, end_value: int
) -> torch.ByteTensor:
    """
    Given a tensor of sequences with shape `(seq_len, num_seqs)`, builds a mask
    of the same shape with values `1` for sequence values appearing prior or at
    the first occurence of `end_value` in each sequence, and `0` for those
    tokens following.
    """
    ends = (sequences == end_value).cpu().numpy()
    # if the end value isn't in a sequence, mark the last token as end
    ends[-1:, :] = True
    # argmax will return the first encountered index
    ends = torch.from_numpy(np.argmax(ends, axis=0))
    # the token index in each sequence
    indices = torch.arange(sequences.size(0), dtype=torch.long)
    # mask each token index prior to and including the end index
    return (indices[:, None] <= ends).type(torch.ByteTensor)


class TrainOutput(NamedTuple):
    sum_loss: float
    num_examples: int
    num_batches: int


def train_epoch(ctx: TrainContext) -> TrainOutput:
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
        the epoch loss and number of batches
    """
    if ctx.seq_len < 4:
        raise ValueError("`seq_len` must be >= 4")
    if 3 + ctx.min_conditional >= ctx.seq_len:
        raise ValueError("`min_conditional` must be < `seq_len - 3`")

    ctx.model.train()

    sum_loss = 0.0

    # NOTE: an epoch is normally a full run over the data, but here it's used
    # to mean some fixed number of training example.

    num_batches = max(1, ctx.epoch_size // ctx.batch_size)
    num_examples = 0

    for _ in range(num_batches):
        # chose some of the sequence starts at random for this batch
        batch_splits = ctx.rng.choice(ctx.batch_indices, ctx.batch_size)
        batch_indices = build_sequence_gather_indices(
            num_tokens=ctx.tokens.size(0),
            splits=batch_splits,
            seq_len=ctx.seq_len,
        )
        # build token batch tensor of shape `(seq_len, batch_size)`
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

        # count each loss term
        num_examples += targets.numel()
        # aggregate to total loss so it can be averaged correctly even if
        # there are batches of different sizes
        sum_loss += loss.item() * targets.numel()

        ctx.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.grad_clip)
        ctx.optimizer.step()

    return TrainOutput(
        sum_loss=sum_loss,
        num_batches=num_batches,
        num_examples=num_examples,
    )


class GeneratorContext(NamedTuple):
    model: torch.nn.Module
    tokenizer: tk.Tokenizer
    special_tokens: SpecialTokens
    max_tokens: int
    rng: np.random.Generator
    temperature: float = 0.9
    temperature_decay: float = 0.7
    temperature_decay_scale: int = 16
    min_uniqueness: float = 0.5
    max_draws: int = 16


def select_uniqueish_token(
    rng: np.random.Generator,
    candidates: List[int],
    sequence: List[int],
    min_uniqueness: float,
) -> int:
    """
    Select an item from a list of candidate items to add to a sequence of items
    such that some uniqueness is preserved.

    Given a list of candidate items, rejects those which are more repetitive
    in the sequence.

    Args:
        rng: the random number generator used to draw
        candidates: list of candidate items
        sequence: current sequence of items to attempt to render unique
        min_uniqueness: decrease the probaility of accepting an item if as its
            frequency in the sequence reaches this value. Below this value,
            the item is always rejected.

    Returns:
        the item index
    """
    if not 0.0 <= min_uniqueness <= 1.0:
        raise ValueError("`min_uniqueness` must be in range [0, 1]")

    unique_set = set(sequence)

    for candidate in candidates:
        is_unique = candidate not in unique_set
        uniqueness = (len(unique_set) + is_unique) / (len(sequence) + 1)

        # at `min_uniqueness == 1.0`, the prob collapses to 0 or 1
        if min_uniqueness == 1.0:
            accept_prob = 1.0 if uniqueness == 1.0 else 0.0
        else:
            accept_prob = max(0, uniqueness - min_uniqueness) / (1.0 - min_uniqueness)

        if rng.random() < accept_prob:
            return candidate

    return None


def generate_seq(
    ctx: GeneratorContext,
    seed: str = None,
    source: str = None,
):
    ctx.model.eval()

    try:
        device = next(ctx.model.parameters()).device
    except StopIteration:
        device = "cuda" if torch.nn.cuda.cuda.is_available() else "cpu"

    src_token_id = ctx.tokenizer.token_to_id(source) if source else None
    if src_token_id is None:
        src_token_id = ctx.special_tokens.unk_id

    input = [
        ctx.special_tokens.src_id,
        src_token_id,
        ctx.special_tokens.bos_id,
    ]
    if seed:
        input += ctx.tokenizer.encode(seed)

    # the tokens to show start after <bos>
    token_ids = list(input)  # list(input[3:])

    input = torch.LongTensor(input).to(device)
    num_generated_tokens = 0

    char_pairs = (
        ("“", "”"),
        ("(", ")"),
    )

    excluded_tokens = {ctx.special_tokens.unk_id}
    for idx in range(ctx.tokenizer.get_vocab_size()):
        token = ctx.tokenizer.id_to_token(idx)
        # only special tokens start with [
        if token.startswith("["):
            excluded_tokens.add(idx)
            continue

        for o, c in char_pairs:
            num_open = token.count(o)
            num_close = token.count(c)
            if num_open != num_close:
                excluded_tokens.add(idx)
                break

    excluded_tokens.discard(ctx.special_tokens.eos_id)
    excluded_tokens = torch.LongTensor(list(sorted(excluded_tokens)))

    with torch.no_grad():  # no tracking history
        while input.size(0) < ctx.max_tokens:
            temperature_progress = min(
                1.0, num_generated_tokens / ctx.temperature_decay_scale
            )
            temperature_decay = math.log(ctx.temperature_decay)
            temperature = ctx.temperature * math.exp(
                temperature_progress * temperature_decay
            )

            # add the (single) batch dimension at index 1
            output = ctx.model(input.unsqueeze(1))
            # select the prediction from the last token, and remove batch dim
            output = output[-1, 0]
            output[excluded_tokens] = -np.inf
            # temperature can squash (expand) the logit values, and the relative
            # probs. is proportional to the differences, which are all smaller
            # (larger) so the things are more (less) evenly distributed
            output = output / temperature
            probs = torch.softmax(output, dim=-1)

            # keep track of uniqueness on the same scale as the temp. decay
            unique_sequence = token_ids[-ctx.temperature_decay_scale :]
            candidates = ctx.rng.choice(
                probs.size(0), ctx.max_draws, p=probs.cpu().numpy()
            )
            token_idx = select_uniqueish_token(
                ctx.rng, candidates, unique_sequence, ctx.min_uniqueness
            )
            # if none of the tokens help make the sequence unique, end it
            if token_idx is None:
                token_idx = ctx.special_tokens.eos_id

            # add it to the input
            input = torch.cat([input, torch.LongTensor([token_idx]).to(device)])

            token_ids.append(token_idx)
            num_generated_tokens += 1

            # when "end of string" is emitted, break early
            if token_idx == ctx.special_tokens.eos_id:
                break

    if token_ids:
        return ctx.tokenizer.decode(token_ids, skip_special_tokens=False)
    else:
        return ""
