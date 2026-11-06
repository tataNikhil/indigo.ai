"""Local token estimates with tiktoken (approximate for non-OpenAI models).

tiktoken downloads its vocabulary file on first use. If that isn't possible
(offline machine, locked-down server) we fall back to ~4 characters per token
so the API keeps working instead of failing every request.
"""

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
