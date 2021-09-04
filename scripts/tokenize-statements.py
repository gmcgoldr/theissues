#!/usr/bin/env python3

"""
Tokenize each `Statement` and encode it as record of the form:

`[sep] src [bos] token ... [eos]`

Where each token is an long int encoding a vocabulary entry. `[sep]`
is the separator, `src` is the source of the statement, `[bos]` is the
beggining of a string and `[eos]` is the end of a string.
"""

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import tokenizers as tk
from typing_extensions import Concatenate

from theissues import utils


def main(
    path_statements: Path,
    path_tokenizer: Path,
    path_sequences: Path,
):
    path_sequences.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))

    statements = []
    with path_statements.open("r") as fio:
        for line in fio:
            if not line.strip():
                continue
            statements.append(utils.Statement(*json.loads(line)))

    print("Tokenizing texts...")
    tokenized = tokenizer.encode_batch(
        list(map(utils.prepare_statement_encoding, statements))
    )
    tokenized = [t.ids for t in tokenized]
    chained = list(itertools.chain.from_iterable(tokenized))

    print("Saving...")
    chained = np.array(chained, dtype=utils.TokenId)
    with path_sequences.open("wb") as fio:
        np.save(fio, chained)
    print(chained[:32])
    print(tokenizer.decode(chained[:32], skip_special_tokens=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("path_sequences", type=Path)
    main(**vars(parser.parse_args()))
