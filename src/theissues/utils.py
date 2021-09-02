import re
import unicodedata
from typing import List

import tokenizers as tk


def normalize_text(text: str) -> str:
    """
    Normalize text to get a connonical respresentation suitable for searching.
    """
    # decompose unicode to reduce the alphabet considered during tokenization,
    # and drop the compatibility character representations
    text = unicodedata.normalize("NFKD", text)
    # use only single space as white space, this also means the new line
    # character is not used in the corpus which is convenient for making line
    # deliminted data
    text = re.sub(r"\s+", " ", text.strip())
    # reserve the square braces for special tokens
    text = text.replace("[", "(")
    text = text.replace("]", ")")
    # remove leading and trailing white space
    text = text.strip()
    return text


class SpecialTokens:
    unk_token = "[UNK]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    src_token = "[SRC]"

    def __init__(self, tokenizer: tk.Tokenizer, validate: bool = False) -> None:
        self.unk_id = tokenizer.token_to_id(self.unk_token)
        self.bos_id = tokenizer.token_to_id(self.bos_token)
        self.eos_id = tokenizer.token_to_id(self.eos_token)
        self.src_id = tokenizer.token_to_id(self.src_token)

        if validate:
            if self.unk_id is None:
                raise ValueError(f"tokenizer does not contain {self.unk_token}")
            if self.bos_id is None:
                raise ValueError(f"tokenizer does not contain {self.bos_token}")
            if self.eos_id is None:
                raise ValueError(f"tokenizer does not contain {self.eos_token}")
            if self.src_id is None:
                raise ValueError(f"tokenizer does not contain {self.src_token}")


def build_entry(
    token_ids: List[int], src_id: int, special_tokens: SpecialTokens
) -> List[int]:
    """
    Build a sequencence entry with a source and special tokens.
    """
    token_ids = [special_tokens.src_id, src_id, special_tokens.bos_id] + token_ids
    token_ids.append(special_tokens.eos_id)
    return token_ids
