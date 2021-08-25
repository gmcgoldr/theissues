import numpy as np
import pytest
import torch

from theissues import training


def test_build_batch_offsets_and_batches():
    rng = np.random.Generator(np.random.PCG64(1234))
    tokens = torch.arange(8)
    # offsets to index 2 and chooses the 2nd batch twice
    batch = training.build_batch(rng=rng, tokens=tokens, seq_len=3, batch_size=2)
    np.testing.assert_equal(batch.tolist(), [[5, 5], [6, 6], [7, 7]])
    # offsets to index 1 and chooses 1st then 2nd batch
    batch = training.build_batch(rng=rng, tokens=tokens, seq_len=3, batch_size=2)
    np.testing.assert_equal(batch.tolist(), [[1, 4], [2, 5], [3, 6]])
    # offsets to index 0 and chooses 1st batch twice
    batch = training.build_batch(rng=rng, tokens=tokens, seq_len=3, batch_size=2)
    np.testing.assert_equal(batch.tolist(), [[0, 0], [1, 1], [2, 2]])
    # offsets to index 0 and chooses 2nd then 1st batch
    batch = training.build_batch(rng=rng, tokens=tokens, seq_len=3, batch_size=2)
    np.testing.assert_equal(batch.tolist(), [[3, 0], [4, 1], [5, 2]])


def test_build_batch_rasise_for_invalid_seq_len():
    rng = np.random.Generator(np.random.PCG64(1234))
    tokens = torch.arange(8)
    # at the limit
    training.build_batch(rng=rng, tokens=tokens, seq_len=4, batch_size=2)
    # above the limit
    with pytest.raises(ValueError):
        training.build_batch(rng=rng, tokens=tokens, seq_len=5, batch_size=2)
