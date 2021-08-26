#!/usr/bin/env python3

import json
import re
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm


def normalize_text(text: str) -> str:
    """
    Normalize text to get a connonical respresentation suitable for searching.
    """
    # lower-case model because user input will have varying levels of
    # case correctness
    text = text.lower()
    # decompose unicode to reduce the alphabet considered during tokenization,
    # and drop the compatibility character representations
    text = unicodedata.normalize("NFKD", text)
    # use only single space as white space, this also means the new line
    # character is not used in the corpus which is convenient for making line
    # deliminted data
    text = re.sub(r"\s+", " ", text.strip())
    # remove leading and trailing white space
    text = text.strip()
    return text


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
        paragraph = normalize_text(paragraph)
        if not paragraph:
            continue
        paragraphs.append(paragraph)

    return paragraphs


def main(
    path_data: Path,
    model_name: str,
    dir_model: Path,
    vocab_size: int,
    max_sentence_chars: int,
    max_token_chars: int,
    split_spaces: bool,
):
    print("Parsing hansards...")
    with (path_data / "hansards.jsonl").open("r") as fio:
        records = [json.loads(l) for l in fio if l.strip()]

    texts = [
        (
            "<pol_{}>".format(r["politician_id"]),
            r["content_en"],
        )
        for r in records
    ]

    paragraphs = []
    sources = []

    for i, (text_src, text) in enumerate(texts):
        try:
            text_paragraphs = build_paragraphs(text)
            paragraphs += text_paragraphs
            sources += [text_src] * len(text_paragraphs)
        except ET.ParseError as e:
            warnings.warn(f"invalid XML in record {i}: {e}")

    # TODO: might want to move functions into a module and test
    # TODO: should verify the sources follow have a single `_` and no new lines
    assert len(sources) == len(paragraphs)

    paragraph_chars = list(map(len, paragraphs))
    low_chars, median_chars, high_chars = np.percentile(paragraph_chars, (5, 50, 95))
    print(
        "Paragraph size in chars: "
        f"{low_chars:.0f} < "
        f"{median_chars:.0f} < "
        f"{high_chars:.0f}"
    )
    print(f"Number of sources: {len(set(sources))}")

    with (path_data / "paragraphs.txt").open("w") as fio:
        fio.write("\n".join(paragraphs))
    with (path_data / "paragraph_sources.txt").open("w") as fio:
        fio.write("\n".join(sources))

    # list of source tokens and the `<src>` seperator so that each sequence
    # can be annodated with a source
    source_tokens = list(sorted(set(sources)))
    source_tokens.insert(0, "<src>")
    source_tokens = ",".join(source_tokens)

    print("Training model...")
    dir_model.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(path_data / "paragraphs.txt"),
        model_prefix=str(dir_model / model_name),
        model_type="UNIGRAM",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        control_symbols=source_tokens,
        max_sentence_length=max_sentence_chars,
        # NOTE: large tokens can capture entire platitudes
        max_sentencepiece_length=max_token_chars,
        # NOTE: non-whitespace delimited tokens are much more domain-specific
        # without much draw-back given SPM's sub-word sampling option
        split_by_whitespace=split_spaces,
        # NOTE: might be some domain-specific words that contain digits
        split_by_unicode_script=False,
    )

    with (dir_model / model_name).with_suffix(".vocab").open("r") as fio:
        vocab = [l.split("\t")[0] for l in fio if l.strip()]

    vocab_chars = list(map(len, vocab))
    vocab_sorted = reversed(sorted(zip(vocab_chars, vocab)))

    print("Longest tokens")
    for i in range(10):
        try:
            _, token = next(vocab_sorted)
        except StopIteration:
            break
        print(f"  {token}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_data", type=Path)
    parser.add_argument("dir_model", type=Path)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--vocab_size", type=int, default=2 ** 15)
    parser.add_argument("--max_sentence_chars", type=int, default=2 ** 10)
    parser.add_argument("--max_token_chars", type=int, default=2 ** 8)
    parser.add_argument("--split_spaces", action="store_true")
    main(**vars(parser.parse_args()))
