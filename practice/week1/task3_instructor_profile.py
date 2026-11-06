"""Task 3: use Instructor to get a UserProfile back from the LLM as typed data.

Instead of parsing free text with regex, we pass `response_model=UserProfile`.
Instructor sends the schema to the model, validates the reply with Pydantic,
and re-asks the model (up to max_retries) if validation fails.

Usage:
    python task3_instructor_profile.py
    python task3_instructor_profile.py "Hi, I'm Priya Sharma, 31, priya@mail.com, PAN bnzps4567q"
"""

import os
import sys

import instructor
from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from user_profile import UserProfile

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

DEFAULT_TEXT = (
    "Hello team, this is Karthik Reddy. I turned 34 last month. "
    "You can reach me at karthik.reddy@example.in and my PAN is afzpk7190k."
)


def extract_profile(text: str) -> UserProfile:
    client = instructor.from_groq(Groq(api_key=API_KEY), mode=instructor.Mode.JSON)
    return client.chat.completions.create(
        model=MODEL,
        response_model=UserProfile,
        temperature=0,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "Extract the user's profile from the text. Do not invent values.",
            },
            {"role": "user", "content": text},
        ],
    )


def main() -> int:
    if not API_KEY:
        print("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT
    print("Input text:", text)
    try:
        profile = extract_profile(text)
    except (ValidationError, instructor.exceptions.InstructorRetryException) as exc:
        print("The model could not produce a valid profile:", exc)
        return 2

    print("\nStructured JSON:")
    print(profile.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
