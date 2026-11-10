"""Local token estimates with tiktoken (approximate for non-OpenAI models)."""

import logging
import math
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("tiktoken vocabulary unavailable; using a character-based estimate")
        return None


def count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return math.ceil(len(text) / 4)
    return len(encoding.encode(text))
