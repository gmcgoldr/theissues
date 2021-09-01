#!/usr/bin/env python3

"""
Build the binary tokenized data for training from the statements.

Creates a file `statements.npy` which is a dense array of packed statements
in the form `<src><src_val><bos>...<eos>` where `...` is a sequence of tokens.
This is encoded into vocabulary entries.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import tokenizers as tk


def build_entry(
    token_ids: List[int], src_val_id: int, bos_id: int, eos_id: int, src_id: int
) -> List[int]:
    token_ids = [src_id, src_val_id, bos_id] + token_ids
    token_ids.append(eos_id)
    return token_ids


def main(
    path_in: Path,
    path_out: Path,
    path_tokenizer: Path,
):
    path_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    src_id = tokenizer.token_to_id("[SRC]")

    if None in {bos_id, eos_id, src_id}:
        raise ValueError("tokenizer must contain: [BOS], [EOS], [SRC]")

    statements = []
    with path_in.open("r") as fio:
        for statement in fio:
            if not statement.strip():
                continue
            source, sequence = json.loads(statement)
            statements.append((source, sequence))

    print("Tokenizing sequences...")
    sequences = [s for _, s in statements]
    tokenized_sequences = tokenizer.encode_batch(sequences)

    print("Counting tokens...")
    counts = defaultdict(int)
    norm = 0
    for sequence in tokenized_sequences:
        for token in sequence.tokens:
            counts[token] += 1
        norm += 1

    counts = list(sorted([(c, t) for t, c in counts.items()]))
    print(f"Unique tokens: {len(counts)}")

    print("Min frequecy:")
    for count, token in counts[:10]:
        print(f"  {token}: {count} ({count / norm * 100:.2f}%)")

    print("Max frequecy:")
    for count, token in counts[-10:]:
        print(f"  {token}: {count} ({count / norm * 100:.2f}%)")

    print("Concatenating...")
    assert len(statements) == len(tokenized_sequences)
    statements = [(s, t.ids) for (s, _), t in zip(statements, tokenized_sequences)]

    concatenated = []
    with path_in.open("r") as fio:
        for source, token_ids in statements:
            src_val_id = tokenizer.token_to_id(source)
            if src_val_id is None:
                warnings.warn(f"unknown source: {source}")
            concatenated += build_entry(
                token_ids=token_ids,
                src_val_id=src_val_id,
                bos_id=bos_id,
                eos_id=eos_id,
                src_id=src_id,
            )

    concatenated = np.array(concatenated, dtype="int64")
    print(tokenizer.decode(concatenated[:32], skip_special_tokens=False))
    with path_out.open("wb") as fio:
        np.save(fio, concatenated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_in", type=Path)
    parser.add_argument("path_out", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    main(**vars(parser.parse_args()))
