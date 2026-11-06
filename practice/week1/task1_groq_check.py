"""Task 1: verify that the Groq API key works and the model answers.

Usage:
    python task1_groq_check.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def main() -> int:
    if not API_KEY:
        print("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    client = Groq(api_key=API_KEY)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: Groq connection OK"},
        ],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    reply = response.choices[0].message.content
    usage = response.usage

    print(f"Model      : {response.model}")
    print(f"Reply      : {reply}")
    print(f"Latency    : {elapsed_ms:.0f} ms")
    if usage:
        print(
            f"Tokens     : prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
