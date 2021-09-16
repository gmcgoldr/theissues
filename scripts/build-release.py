"""
Build the release files.
"""

import json
from pathlib import Path

import numpy as np


def main(
    path_statements: Path, path_embeddings: Path, path_politicians: Path, dir_out: Path
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

    # write the raw binary embeddings to a file as little endian
    with path_embeddings.open("rb") as fio:
        embeddings = np.load(fio)
    embeddings = embeddings.astype("<f")
    with (dir_out / "embeddings.bin").open("wb") as fio:
        embeddings.tofile(fio)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_statements", type=Path)
    parser.add_argument("path_embeddings", type=Path)
    parser.add_argument("path_politicians", type=Path)
    parser.add_argument("dir_out", type=Path)
    main(**vars(parser.parse_args()))
