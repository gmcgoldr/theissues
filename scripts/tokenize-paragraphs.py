#!/usr/bin/env python3

from pathlib import Path
from typing import NamedTuple

import numpy as np
import sentencepiece as spm


def main(
    path_paragraphs: Path,
    path_tokenizer: Path,
    path_tokenized: Path,
    sampling_num: int,
    sampling_alpha: float,
):
    if sampling_alpha is not None and not (0.0 <= sampling_alpha <= 1.0):
        raise ValueError("`sampling_alpha` must be in the range `[0.0, 1.0]`")
    if sampling_num and sampling_alpha is None:
        raise ValueError("`sampling_alpha` must be provided")

    tokenizer = spm.SentencePieceProcessor(model_file=str(path_tokenizer))
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()

    tokens = []

    with path_paragraphs.open("r") as fio:
        for paragraph in fio:
            # in case of trailing empty line, but otherwise the paragraphs
            # should be normalized (stripped)
            if not paragraph.strip():
                continue

            paragraph_tokens = tokenizer.encode(paragraph)
            tokens.append(bos_id)
            tokens += paragraph_tokens
            tokens.append(eos_id)

            for _ in range(sampling_num):
                paragraph_tokens = tokenizer.encode(
                    paragraph, enable_sampling=True, alpha=sampling_alpha
                )
                tokens.append(bos_id)
                tokens += paragraph_tokens
                tokens.append(eos_id)

    tokens = np.array(tokens, dtype="int64")

    with path_tokenized.open("wb") as fio:
        np.save(fio, tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_paragraphs", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("path_tokenized", type=Path)
    parser.add_argument("--sampling_num", type=int)
    parser.add_argument("--sampling_alpha", type=float, default=0.1)
    main(**vars(parser.parse_args()))
