"""Bonus: count tokens with tiktoken and check a prompt fits a context budget.

tiktoken uses OpenAI tokenizers, so for Groq-hosted models the count is an
estimate, but it is good enough for budgeting before sending a request.

Usage:
    python token_budget.py
"""

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def fits_budget(prompt: str, context_window: int, reserved_for_output: int) -> bool:
    return count_tokens(prompt) + reserved_for_output <= context_window


def trim_to_budget(text: str, max_tokens: int) -> str:
    """Keep the first `max_tokens` tokens of `text`."""
    tokens = ENCODING.encode(text)
    return ENCODING.decode(tokens[:max_tokens])


if __name__ == "__main__":
    samples = [
        "Hi",
        "My payment was deducted twice. Please help urgently.",
        "నమస్తే, నా ఆర్డర్ ఎక్కడ ఉంది?",  # non-English text usually costs more tokens
    ]
    for text in samples:
        print(f"{count_tokens(text):>4} tokens | {len(text):>4} chars | {text}")

    long_prompt = "Refund request. " * 500
    print("\nLong prompt tokens:", count_tokens(long_prompt))
    print("Fits 1,000-token window with 200 reserved?", fits_budget(long_prompt, 1000, 200))
    trimmed = trim_to_budget(long_prompt, 800)
    print("After trimming to 800 tokens:", count_tokens(trimmed))
