#!/usr/bin/env python3

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tokenizers as tk
import torch

from theissues import training, utils
from theissues.model import TrainArgs, TransformerModel


def save_model(dir_model: Path, args: TrainArgs, context: training.TrainContext):
    torch.save(context.model.state_dict(), (dir_model / f"state.pt"))
    with (dir_model / f"args.json").open("w") as fio:
        json.dump(args._asdict(), fio, indent="\t")
    with (dir_model / f"model.onnx").open("wb") as fio:
        dummy_input = torch.zeros((context.seq_len, 1), dtype=torch.long).to(
            context.device
        )
        torch.onnx.export(
            model=context.model,
            args=dummy_input,
            f=fio,
            opset_version=10,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )


def main(
    path_statements: Path,
    path_tokenizer: Path,
    dir_model: Path,
    path_log: Path,
    **train_args,
):
    train_args = TrainArgs(**train_args)

    dir_model.mkdir(parents=True, exist_ok=True)
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

    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    special_tokens = utils.SpecialTokens(tokenizer, validate=True)

    with path_statements.open("rb") as fio:
        tokens = np.load(fio)

    tokens = torch.from_numpy(tokens).type(torch.LongTensor)

    rng = np.random.default_rng()

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

    # Adam is a robust choice while other parts of the algorithm and training
    # pipeline are changing
    optimizer = torch.optim.Adam(model.parameters())
    split_token_id = special_tokens.src_id
    split_inidces = training.build_token_splits(tokens, split_token_id)
    train_ctx = training.TrainContext(
        model=model.to(device),
        nvocab=nvocab,
        seq_len=train_args.seq_len,
        min_conditional=train_args.min_conditional,
        device=device,
        tokens=tokens.to(device),
        optimizer=optimizer,
        epoch_size=train_args.batches_per_epoch * train_args.batch_size,
        batch_size=train_args.batch_size,
        batch_indices=split_inidces,
        grad_clip=train_args.grad_clip,
        rng=rng,
    )

    max_epochs = max(1, train_args.max_examples // train_ctx.epoch_size)
    epoch_digits = int(np.log10(max_epochs)) + 1
    epoch_format = f"{{:{epoch_digits}d}}"

    generate_ctx = training.GeneratorContext(
        model=model,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        max_tokens=train_args.seq_len,
    )

    generate_seed_source = (
        (None, "[POL_567]"),  # Trudeau
        (None, "[POL_9243]"),  # O'Toole
        (None, "[POL_10636]"),  # Singh
    )

    try:
        last_loss = np.nan
        for iepoch in range(max_epochs):
            time_start = time.time()
            train_output = training.train_epoch(train_ctx)

            loss = train_output.sum_loss / train_output.num_examples
            diff = loss - last_loss
            last_loss = loss
            time_elapsed = time.time() - time_start
            ns_per_example = time_elapsed * 1e6 / train_output.num_examples
            epoch_str = epoch_format.format(iepoch)

            logging.info(
                f"{epoch_str} / {max_epochs} "
                f"| ns: {ns_per_example:3.0f} "
                f"| loss: {loss:.2e} "
                f"| diff: {diff:+.1e} "
            )

            if iepoch % 24 == 0:
                logging.info("Sample sentences:")
                for seed, source in generate_seed_source:
                    sequence = training.generate_seq(generate_ctx, rng, seed, source)
                    logging.info(f"> {sequence}")
                # save periodically
                save_model(
                    dir_model=dir_model,
                    args=train_args,
                    context=train_ctx,
                )

    except KeyboardInterrupt:
        pass

    logging.info("Sample sentences:")
    for seed, source in generate_seed_source:
        sequence = training.generate_seq(generate_ctx, seed, source)
        logging.info(f"> {sequence}")

    # save the final model
    save_model(dir_model=dir_model, args=train_args, context=train_ctx)


if __name__ == "__main__":
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("dir_model", type=Path)
    parser.add_argument("--path_log", type=Path)

    for field in TrainArgs._fields:
        field_type = TrainArgs._field_types[field]
        field_type = field_type if field_type != bool else strtobool
        default_value = TrainArgs._field_defaults[field]
        parser.add_argument(
            f"--{field}",
            type=field_type,
            default=default_value,
        )

    main(**vars(parser.parse_args()))
