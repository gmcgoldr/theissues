import ctypes
from typing import Iterable, Mapping, NamedTuple, Tuple

import tokenizers as tk


class Statement(NamedTuple):
    source: str
    text: str


TokenId = ctypes.c_int64

Sequence = Iterable[TokenId]

Vocab = Mapping[str, TokenId]


class SpecialTokens:
    unk = "[UNK]"
    sep = "[SEP]"
    bos = "[BOS]"
    eos = "[EOS]"

    def __init__(self, tokenizer: tk.Tokenizer, validate: bool = False) -> None:
        self.unk_id = tokenizer.token_to_id(self.unk)
        self.bos_id = tokenizer.token_to_id(self.bos)
        self.eos_id = tokenizer.token_to_id(self.eos)
        self.sep_id = tokenizer.token_to_id(self.sep)

        if validate:
            if self.unk_id is None:
                raise ValueError(f"tokenizer does not contain {self.unk}")
            if self.bos_id is None:
                raise ValueError(f"tokenizer does not contain {self.bos}")
            if self.eos_id is None:
                raise ValueError(f"tokenizer does not contain {self.eos}")
            if self.sep_id is None:
                raise ValueError(f"tokenizer does not contain {self.sep}")

    @classmethod
    def tokens(cls) -> Tuple[str]:
        return (cls.unk, cls.sep, cls.bos, cls.eos)


def prepare_statement_encoding(statement: Statement) -> str:
    """
    Preparing a statement for encoding. Returns a string of the form:

    `[SEP] source [BOS] text ... [EOS]`

    Args:
        statement: the statement to encode
    """
    return (
        f"{SpecialTokens.sep} "
        f"{statement.source} "
        f"{SpecialTokens.bos} "
        f"{statement.text} "
        f"{SpecialTokens.eos}"
    )
