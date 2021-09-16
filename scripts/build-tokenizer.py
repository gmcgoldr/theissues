#!/usr/bin/env python3

"""
Build the tokenizer. This involves 1) normalizing the data, 2) structuring the
data into sequences, 3) generating the source tokens (list of hansard sources
which currently are politicians).

Outputs the tokenizer and a list of `statements`. A `statement` is a sequence
of text and the ID of the source that issued that sequence. Right now that is
a paragraph and the ID of the policitican who uttered the paragraph.
"""

import json
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
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
        if not paragraph.strip():
            continue
        paragraphs.append(paragraph)

    return paragraphs


def main(
    path_hansards: Path,
    path_statements: Path,
    path_tokenizer: Path,
    vocab_size: int,
    split_lines: bool,
    lower_case: bool,
):
    path_statements.parent.mkdir(parents=True, exist_ok=True)
    path_tokenizer.parent.mkdir(parents=True, exist_ok=True)

    print("Parsing hansards...")
    with path_hansards.open("r") as fio:
        records = [json.loads(l) for l in fio if l.strip()]

    paragraphs = []
    statements = []

    for i, record in enumerate(records):
        text = record["content_en"]
        source = "[POL_{}]".format(record["politician_id"])
        try:
            for paragraph in build_paragraphs(text):
                paragraphs.append(paragraph)
                statements.append(utils.Statement(source, paragraph))
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
    source_tokens = list(sorted(set([s.source for s in statements])))
    print(f"Number of sources: {len(source_tokens)}")

    special_tokens = list(utils.SpecialTokens.tokens()) + source_tokens
    if len(special_tokens) >= vocab_size:
        raise RuntimeError("vocab size must exceed number of special tokens")

    with tempfile.NamedTemporaryFile(mode="w") as fio:
        fio.write("\n".join(paragraphs))

        print("Training...")
        tokenizer = tk.Tokenizer(tk.models.WordPiece(unk_token=utils.SpecialTokens.unk))
        trainer = tk.trainers.WordPieceTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens
        )

        normalizers = [
            # decompose unicode to reduce the alphabet considered during,
            # tokenizations and drop the compatibility character representations
            tk.normalizers.NFKD(),
            # strip leading and trailing whitespace
            tk.normalizers.Strip(),
            # don't allow multiple spaces between tokens
            tk.normalizers.Replace(tk.Regex(r"\s+"), " "),
        ]
        if lower_case:
            normalizers.append(tk.normalizers.Lowercase())
        tokenizer.normalizer = tk.normalizers.Sequence(normalizers)

        if split_lines:
            tokenizer.pre_tokenizer = tk.pre_tokenizers.Sequence(
                [
                    tk.pre_tokenizers.Split(pattern="\n", behavior="removed"),
                    tk.pre_tokenizers.Split(
                        pattern=tk.Regex("[.,]"), behavior="merged_with_previous"
                    ),
                ]
            )
        else:
            tokenizer.pre_tokenizer = tk.pre_tokenizers.Whitespace()

        tokenizer.train([fio.name], trainer)
        tokenizer.save(str(path_tokenizer), pretty=True)

    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_hansards", type=Path)
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("--vocab_size", type=int, default=2 ** 14)
    parser.add_argument("--split_lines", action="store_true")
    parser.add_argument("--lower_case", action="store_true")
    main(**vars(parser.parse_args()))
