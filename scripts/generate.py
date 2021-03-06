#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import tokenizers as tk
import torch

from theissues import training, utils
from theissues.model import TrainArgs, TransformerModel


def main(
    path_tokenizer: Path,
    dir_model: Path,
):
    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    special_tokens = utils.SpecialTokens(tokenizer)

    with (dir_model / "args.json").open("r") as fio:
        train_args = TrainArgs(**json.load(fio))

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

    with (dir_model / "state.pt").open("rb") as fio:
        model.load_state_dict(torch.load(fio))

    rng = np.random.default_rng()

    generate_ctx = training.GeneratorContext(
        model=model,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        max_tokens=train_args.seq_len,
        rng=rng,
    )

    generate_seed_source = (
        (None, "[POL_567]"),  # Trudeau
        (None, "[POL_9243]"),  # O'Toole
        (None, "[POL_10636]"),  # Singh
    )

    for seed, source in generate_seed_source:
        sequence = training.generate_seq(generate_ctx, seed, source)
        print(f"> {sequence}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("dir_model", type=Path)

    main(**vars(parser.parse_args()))
