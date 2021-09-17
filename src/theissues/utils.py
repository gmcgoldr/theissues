import ctypes
from typing import Dict, Iterable, NamedTuple, Tuple

import tokenizers as tk


class Statement(NamedTuple):
    source: str
    text: str


TokenId = ctypes.c_int64

Sequence = Iterable[TokenId]

Vocab = Dict[str, TokenId]


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


def get_id_to_token(vocab: Vocab) -> Dict[TokenId, str]:
    return {i: t for t, i in vocab.items()}


def decode_token_ids(token_ids: Iterable[int], id_to_token: Dict[TokenId, str]) -> str:
    """
    Dedoce a sequence of token ids into its string representation.

    The `Tokenizer`'s `decode` method inserts spaces between tokens and shows
    the `##` sub-word token prefix. This fixes these issues.

    NOTE: if the pre-tokenizer removes whitespace, then this would need to add
    spaces between tokens. But the pre-tokenizer in this code-base is configured
    to not remove any characters such that the text can be fully reconstructed.

    Args:
        token_ids: the ids to decode
        id_to_token: mapping of token ids to the string token

    Returns:
        the decoded text
    """
    tokens = [id_to_token[i] for i in token_ids]
    tokens = [t[2:] if t[:2] == "##" else t for t in tokens]
    return "".join(tokens)
