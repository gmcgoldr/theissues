#!/usr/bin/env python3

import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
import sentencepiece as spm


def build_paragraph_entry(
    tokens: List[int], src_val_id: int, bos_id: int, eos_id: int, src_id: int
) -> List[int]:
    tokens = [src_id, src_val_id, bos_id] + tokens
    tokens.append(eos_id)
    return tokens


def is_iterator_done(iterable: Iterable) -> bool:
    # NOTE: consumes a value
    try:
        next(iterable)
    except StopIteration:
        return True
    return False


def main(
    dir_data: Path,
    path_tokenizer: Path,
    sampling_num: int,
    sampling_alpha: float,
):
    if sampling_alpha is not None and not (0.0 <= sampling_alpha <= 1.0):
        raise ValueError("`sampling_alpha` must be in the range `[0.0, 1.0]`")
    if sampling_num and sampling_alpha is None:
        raise ValueError("`sampling_alpha` must be provided")

    dir_data.mkdir(parents=True, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor(model_file=str(path_tokenizer))
    unk_id = tokenizer.unk_id()
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    src_id = tokenizer.piece_to_id("<src>")

    if unk_id in {bos_id, eos_id, src_id}:
        raise ValueError("tokenizer must contain: <s>, </s>, <src>")

    tokens = []

    with (dir_data / "paragraphs.txt").open("r") as fio_pars, (
        dir_data / "paragraph_sources.txt"
    ).open("r") as fio_srcs:
        for paragraph, source in zip(fio_pars, fio_srcs):
            paragraph = paragraph.strip()
            source = source.strip()

            # in case of trailing empty line, but otherwise the paragraphs
            # should be normalized (stripped)
            if not paragraph:
                continue

            paragraph_tokens = tokenizer.encode(paragraph)
            src_val_id = tokenizer.piece_to_id(source)
            if src_val_id == unk_id:
                warnings.warn(f"unknown source: {source}")

            tokens += build_paragraph_entry(
                tokens=paragraph_tokens,
                src_val_id=src_val_id,
                bos_id=bos_id,
                eos_id=eos_id,
                src_id=src_id,
            )

            for _ in range(sampling_num):
                tokens += build_paragraph_entry(
                    tokens=paragraph_tokens,
                    src_val_id=src_val_id,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    src_id=src_id,
                )

        # TODO: better to store as a jsonl?
        if not is_iterator_done(fio_pars) or not is_iterator_done(fio_srcs):
            raise RuntimeError("inconsistent number of sources and paragraphs")

    tokens = np.array(tokens, dtype="int64")

    with (dir_data / "paragraphs.npy").open("wb") as fio:
        np.save(fio, tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir_data", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("--sampling_num", type=int, default=0)
    parser.add_argument("--sampling_alpha", type=float, default=0.1)
    main(**vars(parser.parse_args()))
