#!/usr/bin/env python3

"""
Evaluate the proximity of statements in pre-defined clusters.
"""

import json
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
):
    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    special_tokens = utils.SpecialTokens(tokenizer)

    print("Reading statements ...")
    with path_statements.open("r") as fio:
        statements = [utils.Statement(*json.loads(l)) for l in fio]

    print("Reading sequences ...")
    with path_sequences.open("rb") as fio:
        sequences = np.load(fio)
    sequence_splits = training.build_sequences_splits(
        torch.from_numpy(sequences), special_tokens.bos_id
    ).numpy()

    with path_embeddings.open("rb") as fio:
        embeddings = np.load(fio)

    num_statements = len(statements)
    if sequence_splits.shape[0] != num_statements:
        raise RuntimeError("sequences don't match statements")
    if embeddings.shape[0] != num_statements:
        raise RuntimeError("embeddings don't match statements")

    print("Building clusters ...")
    topic_groups = clustering.build_topic_groups([s.text for s in statements])

    print("\nCluster sizes:")
    for topic, group in topic_groups.items():
        print(topic, len(group))
        if not group:
            warnings.warn(f"empty topic group: {topic}")

    print("\nEvaluating clusters ...")

    # NOTE: the vector space is effectively a log-probability space for tokens,
    # so the relevant quantities are differencnes. The origin of the space
    # *should* be arbitrary, and cosine distances don't make much sense if it
    # is not centered on the data.
    embeddings = embeddings - np.mean(embeddings, axis=0)[np.newaxis, :]

    rng = np.random.default_rng()

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

    print("\nEuclidean distances:")

    distances_intra = distances_euclid(pairs_intra[:, 0], pairs_intra[:, 1])
    distances_inter = distances_euclid(pairs_inter[:, 0], pairs_inter[:, 1])

    intra_mu = np.mean(distances_intra)
    intra_sig = np.std(distances_intra)
    inter_mu = np.mean(distances_inter)
    inter_sig = np.std(distances_inter)
    separation = (inter_mu - intra_mu) / (inter_sig ** 2 + intra_sig ** 2) ** 0.5

    print(f"Intra distance: {intra_mu:.2e} ± {intra_sig:.2e}")
    print(f"Inter distance: {inter_mu:.2e} ± {inter_sig:.2e}")
    print(f"Separation: {separation:+.2e}")

    print("\nCosine distances:")

    distances_intra = distances_cosine(pairs_intra[:, 0], pairs_intra[:, 1])
    distances_inter = distances_cosine(pairs_inter[:, 0], pairs_inter[:, 1])

    intra_mu = np.mean(distances_intra)
    intra_sig = np.std(distances_intra)
    inter_mu = np.mean(distances_inter)
    inter_sig = np.std(distances_inter)
    separation = (inter_mu - intra_mu) / (inter_sig ** 2 + intra_sig ** 2) ** 0.5

    print(f"Intra distance: {intra_mu:.2e} ± {intra_sig:.2e}")
    print(f"Inter distance: {inter_mu:.2e} ± {inter_sig:.2e}")
    print(f"Separation: {separation:+.2e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_sequences", type=Path)
    parser.add_argument("path_embeddings", type=Path)
    parser.add_argument("path_tokenizer", type=Path)

    main(**vars(parser.parse_args()))
