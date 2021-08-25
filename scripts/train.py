#!/usr/bin/env python3

import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sentencepiece as spm
import torch

from theissues import training
from theissues.model import TransformerModel

TRAIN_ARGS = (
    ("ndims_embed", int, 128),
    ("ndims_trans", int, 128),
    ("nlayers", int, 2),
    ("nheads", int, 2),
    ("dropout", float, 0.2),
    ("seq_len", int, 32),
    ("min_conditional", int, 1),
    ("epoch_size", int, 256),
    ("batch_size", int, 32),
    ("grad_clip", float, 0.5),
    ("max_steps", int, 2 ** 14),
)


def main(path_tokens: Path, path_tokenizer: Path, **train_args):
    tokenizer = spm.SentencePieceProcessor(model_file=str(path_tokenizer))

    with path_tokens.open("rb") as fio:
        tokens = np.load(fio)

    tokens = torch.from_numpy(tokens).type(torch.LongTensor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nvocab = tokenizer.vocab_size()
    model = TransformerModel(
        nvocab=nvocab,
        seq_len=train_args["seq_len"],
        ndims_embed=train_args["ndims_embed"],
        ndims_trans=train_args["ndims_trans"],
        nheads=train_args["nheads"],
        nlayers=train_args["nlayers"],
        dropout=train_args["dropout"],
    )

    # Adam is a robust choice while other parts of the algorithm and training
    # pipeline are changing
    optimizer = torch.optim.Adam(model.parameters())

    train_ctx = training.TrainContext(
        model=model.to(device),
        nvocab=nvocab,
        seq_len=train_args["seq_len"],
        min_conditional=train_args["min_conditional"],
        device=device,
        tokens=tokens.to(device),
        optimizer=optimizer,
        epoch_size=train_args["epoch_size"],
        batch_size=train_args["batch_size"],
        grad_clip=train_args["grad_clip"],
    )

    max_epochs = max(1, train_args["max_steps"] // train_ctx.epoch_size)
    epoch_digits = int(np.log10(max_epochs)) + 1
    epoch_format = f"{{:{epoch_digits}d}}"

    generate_ctx = training.GeneratorContext(
        model=model,
        tokenizer=tokenizer,
        temperature=1e0,
        max_tokens=train_args["seq_len"],
    )

    try:
        last_loss = np.nan
        for iepoch in range(max_epochs):
            time_start = time.time()
            total_loss = training.train_epoch(train_ctx)

            loss = total_loss / train_ctx.epoch_size
            diff = loss - last_loss
            last_loss = loss
            time_elapsed = time.time() - time_start
            ms_per_step = time_elapsed * 1e3 / train_ctx.epoch_size
            epoch_str = epoch_format.format(iepoch)

            print(
                f"{epoch_str} / {max_epochs} "
                f"| ms per step: {ms_per_step:.0f} "
                f"| loss: {loss:.2e} "
                f"| diff: {diff:+.1e} "
            )

            if iepoch % 25 == 0:
                print("Sample sentences:")
                print("> {}".format(training.generate_seq(generate_ctx)))
                print("> {}".format(training.generate_seq(generate_ctx)))
                print("> {}".format(training.generate_seq(generate_ctx)))

    except KeyboardInterrupt:
        pass

    print("\nSample sentences:")
    print("\n{}".format(training.generate_seq(generate_ctx)))
    print("\n{}".format(training.generate_seq(generate_ctx)))
    print("\n{}".format(training.generate_seq(generate_ctx)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_tokens", type=Path)
    parser.add_argument("path_tokenizer", type=Path)

    for field, field_type, default_value in TRAIN_ARGS:
        parser.add_argument(f"--{field}", type=field_type, default=default_value)

    main(**vars(parser.parse_args()))
