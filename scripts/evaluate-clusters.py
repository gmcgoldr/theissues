#!/usr/bin/env python3

"""
Evaluate the proximity of statements in pre-defined clusters.
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import tokenizers as tk
import torch

from theissues import clustering, training, utils


def distances_euclid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the pair-wise distance between each vector in `x` and `y`, where
    the eucledian space is the last axis.
    """
    return np.linalg.norm(x - y, axis=-1)


def distances_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the pair-wise cosine distance between each vector in `x` and `y`,
    where the eucledian space is the last axis.
    """
    dot_product = np.sum(x * y, axis=-1)
    norm = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return 1.0 - dot_product / norm


def main(
    path_statements: Path,
    path_sequences: Path,
    path_embeddings: Path,
    path_tokenizer: Path,
    path_log: Path,
):
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

    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    special_tokens = utils.SpecialTokens(tokenizer)

    logging.info("Reading statements ...")
    with path_statements.open("r") as fio:
        statements = [utils.Statement(*json.loads(l)) for l in fio]

    logging.info("Reading sequences ...")
    with path_sequences.open("rb") as fio:
        sequences = np.load(fio)
    sequence_splits = training.build_sequences_splits(
        torch.from_numpy(sequences), special_tokens.bos_id
    ).numpy()

    with path_embeddings.open("rb") as fio:
        embeddings: np.ndarray = np.load(fio)

    num_statements = len(statements)
    if sequence_splits.shape[0] != num_statements:
        raise RuntimeError("sequences don't match statements")
    if embeddings.shape[0] != num_statements:
        raise RuntimeError("embeddings don't match statements")

    logging.info("Building clusters ...")
    topic_groups = clustering.build_topic_groups([s.text for s in statements])

    logging.info("Cluster sizes:")
    for topic, group in topic_groups.items():
        logging.info("%s: %d", topic, len(group))
        if not group:
            warnings.warn(f"empty topic group: {topic}")

    logging.info("Evaluating clusters ...")

    # NOTE: the vector space is effectively a log-probability space for tokens,
    # so the relevant quantities are differencnes. The origin of the space
    # *should* be arbitrary, and cosine distances don't make much sense if it
    # is not centered on the data.
    embeddings = embeddings - np.mean(embeddings, axis=0)[np.newaxis, :]

    rng: np.random.Generator = np.random.default_rng()

    # NOTE: relying on dict insertion order
    pairs_intra = clustering.build_intra_topic_pairs(
        groups=topic_groups.values(),
        embeddings=embeddings,
        num_pairs=2 ** 15,
        rng=rng,
    )
    pairs_inter = clustering.build_inter_topic_pairs(
        groups=topic_groups.values(),
        embeddings=embeddings,
        num_pairs=2 ** 15,
        rng=rng,
    )

    logging.info("Euclidean distances:")

    distances_intra = distances_euclid(pairs_intra[:, 0], pairs_intra[:, 1])
    distances_inter = distances_euclid(pairs_inter[:, 0], pairs_inter[:, 1])

    intra_mu = np.mean(distances_intra)
    intra_sig = np.std(distances_intra)
    inter_mu = np.mean(distances_inter)
    inter_sig = np.std(distances_inter)
    separation = (inter_mu - intra_mu) / (inter_sig ** 2 + intra_sig ** 2) ** 0.5

    logging.info("Intra distance: %.2e ± %.2e", intra_mu, intra_sig)
    logging.info("Inter distance: %.2e ± %.2e", inter_mu, inter_sig)
    logging.info("Separation: %+.2e", separation)

    logging.info("Cosine distances:")

    distances_intra = distances_cosine(pairs_intra[:, 0], pairs_intra[:, 1])
    distances_inter = distances_cosine(pairs_inter[:, 0], pairs_inter[:, 1])

    intra_mu = np.mean(distances_intra)
    intra_sig = np.std(distances_intra)
    inter_mu = np.mean(distances_inter)
    inter_sig = np.std(distances_inter)
    separation = (inter_mu - intra_mu) / (inter_sig ** 2 + intra_sig ** 2) ** 0.5

    logging.info("Intra distance: %.2e ± %.2e", intra_mu, intra_sig)
    logging.info("Inter distance: %.2e ± %.2e", inter_mu, inter_sig)
    logging.info("Separation: %+.2e", separation)

    logging.info("Cosine neighbours:")
    # want to re-use the same examples everytime this is run
    rng: np.random.Generator = np.random.default_rng(123)

    for topic, group in topic_groups.items():
        if not group:
            continue
        ref_idx = rng.choice(group)
        ref_embedding = embeddings[ref_idx]
        distances = distances_cosine(ref_embedding[np.newaxis, :], embeddings)
        sort_idx = np.argsort(distances)
        logging.info("Topic %s", topic)
        for idx in sort_idx[:3]:
            logging.info("> %s", statements[idx].text)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_sequences", type=Path)
    parser.add_argument("path_embeddings", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("--path_log", type=Path)

    main(**vars(parser.parse_args()))
