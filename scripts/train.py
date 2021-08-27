#!/usr/bin/env python3

import json
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sentencepiece as spm
import torch

from theissues import training
from theissues.model import TransformerModel


class TrainArgs(NamedTuple):
    ndims: int = 256
    nlayers: int = 4
    nheads: int = 4
    dropout: float = 0.0
    seq_len: int = 128
    min_conditional: int = 0
    epoch_size: int = 256
    batch_size: int = 32
    grad_clip: float = 0.5
    max_steps: int = 2 ** 14


def save_model(
    dir_model: Path, name: str, args: TrainArgs, context: training.TrainContext
):
    torch.save(context.model.state_dict(), (dir_model / f"{name}.pt"))
    with (dir_model / f"{name}.json").open("w") as fio:
        json.dump(args._asdict(), fio, indent="\t")
    with (dir_model / f"{name}.onnx").open("wb") as fio:
        dummy_input = torch.zeros((context.seq_len, 1), dtype=torch.long).to(
            context.device
        )
        torch.onnx.export(context.model, dummy_input, fio, opset_version=10)


def main(
    path_statements: Path,
    path_tokenizer: Path,
    dir_model: Path,
    model_name: str,
    path_log: Path,
    **train_args,
):
    train_args = TrainArgs(**train_args)

    dir_model.parent.mkdir(parents=True, exist_ok=True)
    if path_log is not None:
        path_log.parent.mkdir(parents=True, exist_ok=True)

    logging_handlers = [logging.StreamHandler(sys.stdout)]
    if path_log is not None:
        # clear the log file
        with path_log.open("w"):
            pass
        logging_handlers.append(logging.FileHandler(path_log))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=logging_handlers,
    )

    logging.info(json.dumps(train_args._asdict(), indent="\t"))

    tokenizer = spm.SentencePieceProcessor(model_file=str(path_tokenizer))

    with path_statements.open("rb") as fio:
        tokens = np.load(fio)

    tokens = torch.from_numpy(tokens).type(torch.LongTensor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nvocab = tokenizer.vocab_size()
    model = TransformerModel(
        nvocab=nvocab,
        seq_len=train_args.seq_len,
        ndims_embed=train_args.ndims,
        ndims_forward=train_args.ndims,
        nheads=train_args.nheads,
        nlayers=train_args.nlayers,
        dropout=train_args.dropout,
    )

    # Adam is a robust choice while other parts of the algorithm and training
    # pipeline are changing
    optimizer = torch.optim.Adam(model.parameters())
    split_token_id = tokenizer.piece_to_id("<src>")
    split_inidces = training.build_batch_split_indices(tokens, split_token_id)
    train_ctx = training.TrainContext(
        model=model.to(device),
        nvocab=nvocab,
        seq_len=train_args.seq_len,
        min_conditional=train_args.min_conditional,
        device=device,
        tokens=tokens.to(device),
        optimizer=optimizer,
        epoch_size=train_args.epoch_size,
        batch_size=train_args.batch_size,
        batch_indices=split_inidces,
        grad_clip=train_args.grad_clip,
    )

    max_epochs = max(1, train_args.max_steps // train_ctx.epoch_size)
    epoch_digits = int(np.log10(max_epochs)) + 1
    epoch_format = f"{{:{epoch_digits}d}}"

    generate_ctx = training.GeneratorContext(
        model=model,
        tokenizer=tokenizer,
        temperature=1e0,
        max_tokens=train_args.seq_len,
    )
    generate_seed_source = (
        (None, "<pol_567>"),  # Trudeau
        (None, "<pol_9243>"),  # O'Toole
        (None, "<pol_10636>"),  # Singh
    )

    # do an initial save to ensure there are no issues with serialization
    save_model(dir_model=dir_model, name=model_name, args=train_args, context=train_ctx)

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

            logging.info(
                f"{epoch_str} / {max_epochs} "
                f"| ms per step: {ms_per_step:3.0f} "
                f"| loss: {loss:.2e} "
                f"| diff: {diff:+.1e} "
            )

            if iepoch % 24 == 0:
                logging.info("Sample sentences:")
                for seed, source in generate_seed_source:
                    sequence = training.generate_seq(generate_ctx, seed, source)
                    logging.info(f"> {sequence}")

    except KeyboardInterrupt:
        pass

    logging.info("Sample sentences:")
    for seed, source in generate_seed_source:
        sequence = training.generate_seq(generate_ctx, seed, source)
        logging.info(f"> {sequence}")

    # save the final model
    save_model(dir_model=dir_model, name=model_name, args=train_args, context=train_ctx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("dir_model", type=Path)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--path_log", type=Path)

    for field in TrainArgs._fields:
        parser.add_argument(
            f"--{field}",
            type=TrainArgs._field_types[field],
            default=TrainArgs._field_defaults[field],
        )

    main(**vars(parser.parse_args()))
