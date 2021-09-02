#!/usr/bin/env python3

"""
Build the tokenizer and pre-process the data into "statements" which are ready
for tokenization.

Pre-procssing involves normalizing text and mapping statements sources to their
vocabulary token.
"""

import json
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from itertools import repeat
from pathlib import Path
from typing import List

import numpy as np
import tokenizers as tk

from theissues import utils


def build_paragraphs(text: str) -> List[str]:
    """
    Build list of plain text paragraphs from a single HTML-formatted
    hansard statement.

    Extracts the contents of `<p>` elements from the statement.
    """
    paragraphs = []

    # the <br> tags aren't correctly terminated, and it's pretty safe to
    # assume the `<>` characters are only used in HTML so this replacement
    # without parsing is probably fine
    text = re.sub(r"<\s*br\s*>", "<br />", text)

    # need proper XML for parsing, so enclose in a root tag
    data = ET.fromstring("<root>{}</root>".format(text))

    for child in data:
        # not sure what other information might lurk, but for now interested
        # only in the paragraph contents
        if child.tag != "p":
            continue
        # get the string with no HTML markup (links and spans tend to be used)
        paragraph = "".join(child.itertext())
        paragraph = utils.normalize_text(paragraph)
        if not paragraph:
            continue
        paragraphs.append(paragraph)

    return paragraphs


def main(
    path_in: Path,
    path_statements: Path,
    path_tokenizer: Path,
    vocab_size: int,
):
    path_statements.parent.mkdir(parents=True, exist_ok=True)
    path_tokenizer.parent.mkdir(parents=True, exist_ok=True)

    print("Parsing hansards...")
    with path_in.open("r") as fio:
        records = [json.loads(l) for l in fio if l.strip()]

    texts = [
        (
            "[POL_{}]".format(r["politician_id"]),
            r["content_en"],
        )
        for r in records
    ]

    paragraphs = []
    statements = []

    for i, (text_src, text) in enumerate(texts):
        try:
            text_paragraphs = build_paragraphs(text)
            paragraphs += text_paragraphs
            statements += list(zip(repeat(text_src), text_paragraphs))
        except ET.ParseError as e:
            warnings.warn(f"invalid XML in record {i}: {e}")

    with path_statements.open("w") as fio:
        fio.write("\n".join(map(json.dumps, statements)))

    print("Calculating statistics...")
    paragraph_chars = list(map(len, paragraphs))
    low_chars, median_chars, high_chars = np.percentile(paragraph_chars, (5, 50, 95))
    print(
        "Paragraph size in chars: "
        f"{low_chars:.0f} < "
        f"{median_chars:.0f} < "
        f"{high_chars:.0f}"
    )
    source_tokens = list(sorted(set([s for s, _ in statements])))
    print(f"Number of sources: {len(source_tokens)}")

    special_tokens = [
        utils.SpecialTokens.unk_token,
        utils.SpecialTokens.bos_token,
        utils.SpecialTokens.eos_token,
        utils.SpecialTokens.src_token,
    ]
    special_tokens += source_tokens
    if len(special_tokens) >= vocab_size:
        raise RuntimeError("vocab size must exceed number of special tokens")

    with tempfile.NamedTemporaryFile(mode="w") as fio:
        fio.write("\n".join(paragraphs))

        print("Training...")
        tokenizer = tk.Tokenizer(tk.models.WordPiece(unk_token="[UNK]"))
        trainer = tk.trainers.WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )
        tokenizer.pre_tokenizer = tk.pre_tokenizers.Whitespace()
        tokenizer.train([fio.name], trainer)
        tokenizer.save(str(path_tokenizer), pretty=True)

    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_in", type=Path)
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("--vocab_size", type=int, default=2 ** 13)
    main(**vars(parser.parse_args()))
