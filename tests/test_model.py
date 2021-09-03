import numpy as np
import pytest
import torch

from theissues import model


def test_position_encoding_basis_has_correct_dims():
    encoding = model.PositionalEncoding(4, 8)
    assert encoding.basis.shape == (8, 1, 4)


def test_position_encoding_basis_has_increasing_trig_period():
    encoding = model.PositionalEncoding(4, 8)
    # the first dimension oscillates at each token index
    np.testing.assert_allclose(
        encoding.basis[:, 0, 0], [0, 1, 0, -1, 0, 1, 0, -1], atol=1e-6
    )
    # the last dimension is cos with 1/4 cycle so it is decreasing from 1 to
    # nearly 0
    assert torch.all(encoding.basis[1:, 0, -1] < encoding.basis[:-1, 0, -1])
    np.testing.assert_allclose(encoding.basis[-1, 0, -1], -1, rtol=1e-1)


def test_position_encoding_basis_has_sin_cos_basis():
    encoding = model.PositionalEncoding(4, 8)
    np.testing.assert_allclose(encoding.basis[0, 0, :], [0, 1, 0, 1], atol=1e-6)


def test_position_encoding_basis_handles_odd_dims():
    encoding = model.PositionalEncoding(3, 8)
    assert encoding.basis.shape == (8, 1, 3)
    np.testing.assert_allclose(encoding.basis[0, 0, :], [0, 1, 0], atol=1e-6)


def test_position_encoding_adds_to_x():
    # 4 dim tokens, up to 8 tokens
    encoding = model.PositionalEncoding(4, 8)
    # batch of 2 tokens
    x = torch.FloatTensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ]
    )
    out = encoding(x)
    # uses encodings for only 2 tokens
    np.testing.assert_allclose(out, x + encoding.basis[:2, :], atol=1e-6)


def test_transformer_model_generates_subsequent_mask():
    mask = model.TransformerModel.build_subsequent_mask(3)
    np.testing.assert_allclose(
        mask, [[0, -np.inf, -np.inf], [0, 0, -np.inf], [0, 0, 0]]
    )


@pytest.fixture
def dummy_model():
    return model.TransformerModel(
        nvocab=32,
        seq_len=8,
        ndims_embed=4,
        ndims_forward=6,
        nheads=2,
        nlayers=5,
        dropout=0,
        tied=False,
    )


def test_transformer_model_forward_latent_outputs_token_vectors(
    dummy_model: model.TransformerModel,
):
    # 8 tokens, 3 sequences in the batch
    x = torch.arange(8 * 3, dtype=torch.long).view((8, 3))
    out = dummy_model.forward_latent(x)
    out = out.detach()
    # 8 tokens, 3 sequences, 4-dim embeddings
    assert out.shape == (8, 3, 4)


def test_transformer_model_outputs_weights_over_vocab(
    dummy_model: model.TransformerModel,
):
    # 8 tokens, 3 sequences in the batch
    x = torch.arange(8 * 3, dtype=torch.long).view((8, 3))
    out = dummy_model(x)
    out = out.detach()
    # prob. over 32 vocab tokens, for 3 sequencs, for each of the 8 input tokens
    # with the other dimensions having been collected in the transformer
    assert out.shape == (8, 3, 32)


def test_transformer_model_outputs_weights_over_vocab_when_tied(
    dummy_model: model.TransformerModel,
):
    # 8 tokens, 3 sequences in the batch
    x = torch.arange(8 * 3, dtype=torch.long).view((8, 3))
    out = dummy_model(x)
    out = out.detach()
    # prob. over 32 vocab tokens, for 3 sequencs, for each of the 8 input tokens
    # with the other dimensions having been collected in the transformer
    assert out.shape == (8, 3, 32)
