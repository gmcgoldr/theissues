#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import sentencepiece as spm


def main(path_paragraphs: Path, path_tokenizer: Path, path_tokenized: Path):
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

            paragraph = tokenizer.encode(paragraph)

            tokens.append(bos_id)
            tokens += paragraph
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
    main(**vars(parser.parse_args()))
