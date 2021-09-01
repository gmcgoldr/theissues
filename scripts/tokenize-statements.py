#!/usr/bin/env python3

"""
Build the binary tokenized data for training from the statements.

Creates a file `statements.npy` which is a dense array of packed statements
in the form `<src><src_val><bos>...<eos>` where `...` is a sequence of tokens.
This is encoded into vocabulary entries.
"""

import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm


def build_entry(
    tokens: List[int], src_val_id: int, bos_id: int, eos_id: int, src_id: int
) -> List[int]:
    tokens = [src_id, src_val_id, bos_id] + tokens
    tokens.append(eos_id)
    return tokens


def main(
    path_in: Path,
    path_out: Path,
    path_tokenizer: Path,
    sampling_num: int,
    sampling_alpha: float,
):
    if sampling_alpha is not None and not (0.0 <= sampling_alpha <= 1.0):
        raise ValueError("`sampling_alpha` must be in the range `[0.0, 1.0]`")
    if sampling_num and sampling_alpha is None:
        raise ValueError("`sampling_alpha` must be provided")

    tokenizer = spm.SentencePieceProcessor(model_file=str(path_tokenizer))
    unk_id = tokenizer.unk_id()
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    src_id = tokenizer.piece_to_id("<src>")

    if unk_id in {bos_id, eos_id, src_id}:
        raise ValueError("tokenizer must contain: <s>, </s>, <src>")

    tokens = []

    with path_in.open("r") as fio:
        for statement in fio:
            if not statement.strip():
                continue
            source, sequence = json.loads(statement)

            paragraph_tokens = tokenizer.encode(sequence)
            src_val_id = tokenizer.piece_to_id(source)
            if src_val_id == unk_id:
                warnings.warn(f"unknown source: {source}")

            tokens += build_entry(
                tokens=paragraph_tokens,
                src_val_id=src_val_id,
                bos_id=bos_id,
                eos_id=eos_id,
                src_id=src_id,
            )

            for _ in range(sampling_num):
                tokens += build_entry(
                    tokens=paragraph_tokens,
                    src_val_id=src_val_id,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    src_id=src_id,
                )

    tokens = np.array(tokens, dtype="int64")

    with path_out.open("wb") as fio:
        np.save(fio, tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_in", type=Path)
    parser.add_argument("path_out", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("--sampling_num", type=int, default=0)
    parser.add_argument("--sampling_alpha", type=float, default=0.1)
    main(**vars(parser.parse_args()))
