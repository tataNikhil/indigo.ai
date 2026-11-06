"""Why regex parsing breaks on LLM output, and why schema validation does not.

LLM replies are non-deterministic: the same question can come back as
"Priority: High", "priority - HIGH", a JSON blob, or a sentence. A regex that
worked yesterday silently returns None today. A Pydantic model either gives
you a correctly typed object or raises a clear error you can handle.

Usage:
    python regex_vs_schema.py
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError


class Ticket(BaseModel):
    priority: Literal["low", "medium", "high"]
    refund_amount: float


# Five plausible replies an LLM might give to the same prompt.
REPLIES = [
    "Priority: high\nRefund amount: 499",
    "priority - HIGH, refund of Rs. 499.00",
    "I'd rate this as high priority. The refund should be ₹499.",
    '{"priority": "high", "refund_amount": "499"}',
    '{"priority": "urgent", "refund_amount": "four hundred"}',
]

REGEX = re.compile(r"Priority:\s*(\w+)\s*\nRefund amount:\s*([\d.]+)")


def parse_with_regex(text: str):
    match = REGEX.search(text)
    return match.groups() if match else None


def parse_with_schema(text: str):
    try:
        return Ticket.model_validate(json.loads(text))
    except json.JSONDecodeError:
        return "not JSON -> ask the model again"
    except ValidationError as exc:
        return f"rejected: {exc.error_count()} validation error(s)"


if __name__ == "__main__":
    for reply in REPLIES:
        print(f"\nReply : {reply!r}")
        print(f"Regex : {parse_with_regex(reply)}")
        print(f"Schema: {parse_with_schema(reply)}")
