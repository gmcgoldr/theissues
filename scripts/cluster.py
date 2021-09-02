#!/usr/bin/env python3

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tokenizers as tk
import torch

from theissues import training
from theissues.model import TrainArgs, TransformerModel


def main(
    path_statements: Path,
    path_tokenizer: Path,
    dir_model: Path,
):
    with (dir_model / "args.json").open("r") as fio:
        train_args = TrainArgs(**json.load(fio))

    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))

    with path_statements.open("rb") as fio:
        tokens = np.load(fio)

    tokens = torch.from_numpy(tokens).type(torch.LongTensor)

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

    with (dir_model / "state.pt").open("rb") as fio:
        model.load_state_dict(torch.load(fio))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("dir_model", type=Path)

    main(**vars(parser.parse_args()))
