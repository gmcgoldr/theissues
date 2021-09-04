#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import tokenizers as tk
import torch
from tqdm import tqdm

from theissues import training, utils
from theissues.model import TrainArgs, TransformerModel


def main(
    path_sequences: Path,
    path_tokenizer: Path,
    dir_model: Path,
    path_embeddings: Path,
):
    path_embeddings.parent.mkdir(parents=True, exist_ok=True)

    with (dir_model / "args.json").open("r") as fio:
        train_args = TrainArgs(**json.load(fio))

    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    special_tokens = utils.SpecialTokens(tokenizer, validate=True)

    with path_sequences.open("rb") as fio:
        sequences = np.load(fio)
    sequences = torch.from_numpy(sequences).type(torch.LongTensor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nvocab = tokenizer.get_vocab_size()
    model = TransformerModel(
        nvocab=nvocab,
        seq_len=train_args.seq_len,
        ndims_embed=train_args.ndims,
        ndims_forward=train_args.ndims,
        nheads=train_args.nheads,
        nlayers=train_args.nlayers,
        dropout=train_args.dropout,
        tied=train_args.tied_weights,
    )
    model.eval()

    model = model.to(device)
    sequences = sequences.to(device)

    with (dir_model / "state.pt").open("rb") as fio:
        model.load_state_dict(torch.load(fio))

    # start sequences at the BOS tokens (not SRC as in training, this ignores
    # the sequence source)
    seq_splits = training.build_sequences_splits(sequences, special_tokens.bos_id)
    print(f"Number of sequences: {seq_splits.size(0)}")

    batch_size = 2 ** 10
    num_batches = math.ceil(seq_splits.size(0) / batch_size)

    sequence_embeds = []

    with torch.no_grad():
        for ibatch in tqdm(range(num_batches)):
            istart = ibatch * batch_size
            batch_splits = seq_splits[istart : istart + batch_size]
            batch_indices = training.build_sequence_gather_indices(
                num_tokens=sequences.size(0),
                splits=batch_splits,
                seq_len=train_args.seq_len,
            )
            batch_sequences = sequences[batch_indices]
            batch_mask = training.build_sequence_mask_after(
                batch_sequences, special_tokens.eos_id
            )

            out = model.forward_latent(batch_sequences)
            # apply the mask so that information gather for tokens past the
            # first EOS isn't considered
            batch_mask = batch_mask.to(out.device)
            out = out * batch_mask[:, :, None]
            # take the mean of the token outputs up to and including EOS
            out = torch.sum(out, dim=0) / torch.sum(batch_mask, dim=0)[:, None]

            sequence_embeds += out.tolist()

    with path_embeddings.open("wb") as fio:
        np.save(fio, sequence_embeds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_sequences", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("dir_model", type=Path)
    parser.add_argument("path_embeddings", type=Path)

    main(**vars(parser.parse_args()))
