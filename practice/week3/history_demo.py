"""Week 3 pre-class task: send conversation history to Groq.

Part 1 - stateless: every call is independent, so the model forgets your name.
Part 2 - stateful: we keep a list of messages and resend it on every call.
Part 3 - two session IDs, two separate histories that never mix.

Uses the key from backend/.env, so no extra setup is needed.

Usage:
    python history_demo.py
"""

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[2] / "backend" / ".env")
load_dotenv()  # a local .env still works too

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SYSTEM = {"role": "system", "content": "You are a helpful assistant. Answer in one short sentence."}


def ask(client: Groq, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=MODEL, temperature=0, messages=messages)
    return response.choices[0].message.content.strip()


def stateless(client: Groq) -> None:
    print("=== Part 1: stateless (no history) ===")
    for question in ["Hi, my name is Nikhil and my order ID is A1023.", "What is my order ID?"]:
        answer = ask(client, [SYSTEM, {"role": "user", "content": question}])
        print(f"You : {question}\nBot : {answer}\n")


def stateful(client: Groq) -> None:
    print("=== Part 2: stateful (history is resent every time) ===")
    history = [SYSTEM]
    for question in ["Hi, my name is Nikhil and my order ID is A1023.", "What is my order ID?"]:
        history.append({"role": "user", "content": question})
        answer = ask(client, history)
        history.append({"role": "assistant", "content": answer})
        print(f"You : {question}\nBot : {answer}\n")
    print(f"(the second call sent {len(history) - 1} messages to the model)\n")


def separate_sessions(client: Groq) -> None:
    print("=== Part 3: session IDs keep chats apart ===")
    sessions: dict[str, list[dict]] = {}

    def chat(session_id: str, text: str) -> str:
        history = sessions.setdefault(session_id, [SYSTEM])
        history.append({"role": "user", "content": text})
        answer = ask(client, history)
        history.append({"role": "assistant", "content": answer})
        return answer

    alice, bob = str(uuid.uuid4()), str(uuid.uuid4())
    chat(alice, "My favourite colour is green.")
    chat(bob, "My favourite colour is orange.")
    print(f"Session {alice[:8]} -> {chat(alice, 'What is my favourite colour?')}")
    print(f"Session {bob[:8]} -> {chat(bob, 'What is my favourite colour?')}")


def main() -> int:
    if not API_KEY:
        print("GROQ_API_KEY is not set. Add it to backend/.env.")
        return 1
    client = Groq(api_key=API_KEY)
    stateless(client)
    stateful(client)
    separate_sessions(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
