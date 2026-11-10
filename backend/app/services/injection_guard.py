"""Cheap, deterministic checks that run on every message, independent of the LLM."""

import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"you are now\b",
    r"act as (an? )?(admin|developer|system)",
    r"(reveal|print|show) (me )?(your|the) (system prompt|instructions)",
    r"(set|mark|classify) (this|the) (risk|priority|intent) (as|to)",
    r"respond (only )?with (the )?(json|following)",
]
_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]

# Basic PII hints: card-like numbers, Indian PAN, Aadhaar-like 12 digits.
PII_PATTERNS = [
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
]


def detect_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _COMPILED)


def detect_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII_PATTERNS)
