"""
Build the release files.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tokenizers as tk
import torch

from theissues import utils
from theissues.model import TrainArgs


def main(
    path_statements: Path,
    path_embeddings: Path,
    path_politicians: Path,
    path_tokenizer: Path,
    dir_model: Path,
    dir_out: Path,
):
    dir_out.mkdir(exist_ok=True, parents=True)

    # convert the statements to a flat json list
    with path_statements.open("r") as fio:
        statements = [json.loads(l) for l in fio if l.strip()]
    with (dir_out / "statements.json").open("w") as fio:
        json.dump(statements, fio)

    seen_politicians = {int(s[0][1:-1].split("_")[1]) for s in statements}

    # convert the politicians to a flat json mapping
    with path_politicians.open("r") as fio:
        politicians = [json.loads(l) for l in fio if l.strip()]
    politicians = {
        p["id"]: p["name"] for p in politicians if p["id"] in seen_politicians
    }
    with (dir_out / "politicians.json").open("w") as fio:
        json.dump(politicians, fio)

    # write the raw binary embeddings
    with path_embeddings.open("rb") as fio:
        embeddings = np.load(fio)
    # ensure little endian 32-bit float
    embeddings = embeddings.astype("<f")
    with (dir_out / "embeddings.bin").open("wb") as fio:
        embeddings.tofile(fio)

    # write the raw binary word vectors
    with (dir_model / "args.json").open("r") as fio:
        train_args = TrainArgs(**json.load(fio))
    with (dir_model / "state.pt").open("rb") as fio:
        model_state = torch.load(fio)
    # want the word vectors used to decode because these are the vectors which
    # couple words in a sequence to the transformer latent state
    if train_args.tied_weights:
        word_vectors = model_state["embeddings.weight"]
    else:
        word_vectors = model_state["decoder.weight"]
    # ensure little endian 32-bit float
    word_vectors = word_vectors.cpu().numpy().astype("<f")
    with (dir_out / "word_vectors.bin").open("wb") as fio:
        word_vectors.tofile(fio)
    with (dir_out / "word_vectors_shape.json").open("w") as fio:
        json.dump(word_vectors.shape, fio)

    # copy the model
    shutil.copy(str(dir_model / "model.onnx"), str(dir_out / "model.onnx"))

    # write the vocabulary
    tokenizer = tk.Tokenizer.from_file(str(path_tokenizer))
    vocab = tokenizer.get_vocab()
    vocab = sorted(vocab.items(), key=lambda i: (i[1], i[0]))
    vocab = {t: i for t, i in vocab}
    with (dir_out / "vocabulary.json").open("w") as fio:
        json.dump(vocab, fio, indent="\t")
    id_to_token = utils.get_id_to_token(vocab)
    with (dir_out / "id_to_token.json").open("w") as fio:
        json.dump(id_to_token, fio, indent="\t")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_embeddings", type=Path)
    parser.add_argument("path_politicians", type=Path)
    parser.add_argument("path_tokenizer", type=Path)
    parser.add_argument("dir_model", type=Path)
    parser.add_argument("dir_out", type=Path)
    main(**vars(parser.parse_args()))
