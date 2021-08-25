#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from theissues import training
from theissues.model import TransformerModel


def main(path_tokens: Path, path_tokenizer: Path):
    tokenizer = spm.SentencePieceProcessor(model_file=str(path_tokenizer))

    with path_tokens.open("rb") as fio:
        tokens = np.load(fio)

    tokens = torch.from_numpy(tokens).type(torch.LongTensor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nvocab = tokenizer.vocab_size()
    ndims_embed = 128
    ndims_trans = 128
    nlayers = 2
    nheads = 2
    dropout = 0.2
    seq_len = 32
    model = TransformerModel(
        nvocab=nvocab,
        seq_len=seq_len,
        ndims_embed=ndims_embed,
        ndims_trans=ndims_trans,
        nheads=nheads,
        nlayers=nlayers,
        dropout=dropout,
    )

    # Adam is a robust choice while other parts of the algorithm and training
    # pipeline are changing
    optimizer = torch.optim.Adam(model.parameters())

    train_ctx = training.TrainContext(
        model=model.to(device),
        nvocab=nvocab,
        seq_len=seq_len,
        device=device,
        tokens=tokens.to(device),
        optimizer=optimizer,
        epoch_size=256,
        batch_size=32,
        grad_clip=0.5,
    )

    max_steps = 2 ** 14
    max_epochs = max_steps // train_ctx.epoch_size
    epoch_digits = int(np.log10(max_epochs)) + 1
    epoch_format = f"{{:{epoch_digits}d}}"

    generate_ctx = training.GeneratorContext(
        model=model,
        tokenizer=tokenizer,
        temperature=1e0,
        max_tokens=seq_len,
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
    main(**vars(parser.parse_args()))
