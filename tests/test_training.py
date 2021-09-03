import numpy as np
import pytest
import torch

from theissues import training


def test_build_token_splits_returns_indices():
    tokens = torch.LongTensor([1, 2, 3, 1, 2])
    indices = training.build_token_splits(tokens, 2)
    np.testing.assert_equal(indices.tolist(), [1, 4])


def test_build_token_splits_handles_out_of_range():
    tokens = torch.LongTensor([1, 2, 3, 1, 2])
    indices = training.build_token_splits(tokens, 4)
    np.testing.assert_equal(indices.tolist(), [])


def test_token_split_gather_indices_gathers_sequences():
    num_tokens = 3
    indices = training.build_token_split_gather_indices(num_tokens, [0, 1], 4)
    assert indices.tolist() == [
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 1],
    ]


def test_build_sequence_mask_after_masks_tokens():
    sequences = torch.LongTensor(
        [
            [0, 0],
            [1, 1],
            [3, 2],
            [3, 2],
            [2, 2],
        ]
    )
    mask = training.build_sequence_mask_after(sequences, 3)
    # masks after the first occurrence of 3, or nothing if there is no 3
    assert mask.tolist() == [
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
    ]


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
