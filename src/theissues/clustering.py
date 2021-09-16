import re
from typing import List, Mapping, Tuple

import numpy as np

TopicGroups = Mapping[str, List[int]]


def calculate_pca(x: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run principal component analysis on the tensor `x` with shape `(m, n)` where
    `m` is the number of data points, and `n` is the number of dimensions in
    the space of `x`.

    Returns the `bias` and `basis` vector which can be used to transform data
    to the PCA space with `q` dimensions using:

    `np.dot(x, basis)`

    or for a centered space:

    `np.dot(x + bias, basis)`

    The `bias` tensor mean centers data in the space of `x`. The columns of the
    `basis` tensor are eigenvectors in the space of `x`.

    Args:
        x: the data points from which to calculate the PCA, with shape `(n, m)`
            where `n` is the number of points and `m` is the dimensionality of
            the input space
        q: the number of dimenisions of the transformed space

    Returns:
        the bias vector, and basis matrix
    """
    if q > x.shape[1]:
        raise ValueError("`q` must be <= the dimensions of `x`")

    bias = -np.mean(x, axis=0)
    # note: centering doesn't affect this calculation
    std = np.maximum(1e-6, np.std(x, axis=0))
    # copies x
    x = (x + bias) / std

    # esimate the covariance
    cov = np.dot(x.T, x) / x.shape[0]
    # extract the eingevectors and sorty by descending variance
    scale, basis = np.linalg.eigh(cov)
    sorted_idx = np.argsort(scale)[::-1]
    basis = basis[:, sorted_idx]

    # keep only `ndims` top axes in the PCA space
    basis = basis[:, :q]

    return bias, basis


def build_topic_groups(texts: List[str]) -> TopicGroups:
    """
    Build a mapping of cluster names to statement indices that fall into that
    cluster. This is based on a fixed list of expected clusters. The statements
    are identified using keywords.
    """

    clusters = {
        "CERB": [],
        "Residential Schools": [],
        "Carbon Tax": [],
        "Vaccine": [],
        "Reopening": [],
    }

    for i, text in enumerate(texts):
        if re.search(r"\bCERB\b", text):
            clusters["CERB"].append(i)
        if re.search(
            r"\bcanadian emergency response benefit\b", text, flags=re.IGNORECASE
        ):
            clusters["CERB"].append(i)
        if re.search(r"\bresidential schools?\b", text, flags=re.IGNORECASE):
            clusters["Residential Schools"].append(i)
        if re.search(r"\btruth and reconciliation\b", text, flags=re.IGNORECASE):
            clusters["Residential Schools"].append(i)
        if re.search(r"\bcarbon tax\b", text, flags=re.IGNORECASE):
            clusters["Carbon Tax"].append(i)
        if re.search(r"\bprice on carbon\b", text, flags=re.IGNORECASE):
            clusters["Carbon Tax"].append(i)
        if re.search(r"\bvaccines?\b", text, flags=re.IGNORECASE):
            clusters["Vaccine"].append(i)
        if re.search(r"\bPfizer\b", text):
            clusters["Vaccine"].append(i)
        if re.search(r"\bModerna\b", text):
            clusters["Vaccine"].append(i)
        if re.search(r"\bAstraZeneca\b", text):
            clusters["Vaccine"].append(i)
        if re.search(r"\breopen(?:ing)?\b", text, flags=re.IGNORECASE):
            clusters["Reopening"].append(i)

    return clusters


def build_intra_topic_pairs(
    groups: List[List[int]],
    embeddings: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build an array with pairs of embedding vectors for statements from the same
    group.

    Args:
        groups: list of groups, each is a list of indices in the embedding array
            of those embeddings in the same group
        embeddings: sequence embeddings with shape `(num_sequences, num_dims)`
        num_pairs: draw this many pairs from *each* group
        rng: random number generator to use for drawing pairs

    Returns:
        array of embeddings with shape `(num_groups * num_pairs, 2, num_dims)`
        where `axis=1` is over the two embeddings in a pair
    """
    ngroups = len(groups)
    ndims = embeddings.shape[-1]
    pairs = np.zeros((ngroups, num_pairs * 2, ndims), dtype=embeddings.dtype)

    for igroup, group in enumerate(groups):
        pairs[igroup] = embeddings[rng.choice(group, 2 * num_pairs)]

    pairs = pairs.reshape((ngroups * num_pairs, 2, ndims))

    return pairs


def build_inter_topic_pairs(
    groups: List[List[int]],
    embeddings: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build an array with pairs of embedding vectors for statements from different
    groups.

    Args:
        groups: list of groups, each is a list of indices in the embedding array
            of those embeddings in the same group
        embeddings: sequence embeddings with shape `(num_sequences, num_dims)`
        num_pairs: draw this many pairs for *each* group
        rng: random number generator to use for drawing pairs

    Returns:
        array of embeddings with shape `(num_groups * num_pairs, 2, num_dims)`
        where `axis=1` is over the two embeddings in a pair
    """
    ngroups = len(groups)
    ndims = embeddings.shape[-1]
    pairs = np.zeros((ngroups, num_pairs, 2, ndims), dtype=embeddings.dtype)

    for igroup, group in enumerate(groups):
        group_other = sum([g for i, g in enumerate(groups) if i != igroup], [])
        indices_0 = rng.choice(group, num_pairs)
        indices_1 = rng.choice(group_other, num_pairs)
        pairs[igroup, :, 0] = embeddings[indices_0]
        pairs[igroup, :, 1] = embeddings[indices_1]

    pairs = pairs.reshape((ngroups * num_pairs, 2, ndims))

    return pairs
