import numpy as np
import pytest
import torch

from theissues import training


def test_build_batch_split_indices_returns_indices():
    tokens = torch.LongTensor([1, 2, 3, 1, 2])
    indices = training.build_batch_split_indices(tokens, 2)
    np.testing.assert_equal(indices.tolist(), [1, 4])


def test_build_batch_split_indices_handles_out_of_range():
    tokens = torch.LongTensor([1, 2, 3, 1, 2])
    indices = training.build_batch_split_indices(tokens, 4)
    np.testing.assert_equal(indices.tolist(), [])


def test_build_batch_indices_returns_contiguous_sequences_at_split():
    rng = np.random.Generator(np.random.PCG64(123))
    tokens = torch.arange(8)
    split_idxs = torch.LongTensor([1])
    indices = training.build_batch_indices(rng, tokens, split_idxs, 3, 2)
    # can select only sequences starting at `1`, so it will return both
    # sequences there, of length 3 (note seq along dim 0)
    np.testing.assert_equal(indices.tolist(), [[1, 1], [2, 2], [3, 3]])


def test_build_batch_indices_returns_in_bounds_with_wraparound():
    rng = np.random.Generator(np.random.PCG64(123))
    tokens = torch.arange(8)
    split_idxs = torch.LongTensor([7])
    indices = training.build_batch_indices(rng, tokens, split_idxs, 3, 2)
    np.testing.assert_equal(indices.tolist(), [[7, 7], [0, 0], [1, 1]])


def test_build_batch_indices_returns_in_bounds_with_wraparound():
    rng = np.random.Generator(np.random.PCG64(123))
    tokens = torch.arange(8)
    split_idxs = torch.LongTensor([1, 2])
    indices = training.build_batch_indices(rng, tokens, split_idxs, 3, 2)
    np.testing.assert_equal(indices.tolist(), [[1, 2], [2, 3], [3, 4]])


def test_select_uniqueish_tokens_raises_for_invalid_min():
    rng = np.random.Generator(np.random.PCG64(123))
    sequence = ["a", "b"]
    candidates = ["a"]
    # doesn't raise
    training.select_uniqueish_token(rng, candidates, sequence, 0.0)
    training.select_uniqueish_token(rng, candidates, sequence, 1.0)
    # raises
    with pytest.raises(ValueError):
        training.select_uniqueish_token(rng, candidates, sequence, -0.01)
    with pytest.raises(ValueError):
        training.select_uniqueish_token(rng, candidates, sequence, 1.01)


def test_select_uniqueish_tokens_rejects_below_min():
    rng = np.random.Generator(np.random.PCG64(123))
    sequence = ["a", "b"]
    candidates = ["a"]
    # can't select the candidate as it will cause "a" to appear 0.667 > 0.5
    assert training.select_uniqueish_token(rng, candidates, sequence, 0.5) is None


def test_select_uniqueish_tokens_accepts_new_token_above_min():
    rng = np.random.Generator(np.random.PCG64(123))
    sequence = ["a", "b"]
    candidates = ["a", "b"]
    # has 66% chance of accepting, with RNG this happens on 2nd try
    assert training.select_uniqueish_token(rng, candidates, sequence, 0.0) == "b"


def test_select_uniqueish_tokens_accepts_new_token_with_min_1():
    rng = np.random.Generator(np.random.PCG64(123))
    sequence = ["a", "b"]
    candidates = ["c"]
    assert training.select_uniqueish_token(rng, candidates, sequence, 1.0) == "c"
