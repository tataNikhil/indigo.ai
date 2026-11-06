"""Task 2: validate user profiles with Pydantic and handle ValidationError.

Usage:
    python task2_pydantic_validation.py
"""

from pydantic import ValidationError

from user_profile import UserProfile

SAMPLES = [
    # Valid. Note type coercion: age "29" (str) becomes 29 (int),
    # and the lowercase PAN is normalised to uppercase.
    {"name": "ravi kumar", "age": "29", "email": "ravi@example.com", "pan": "abcpk1234f"},
    # Invalid: underage, bad email, bad PAN.
    {"name": "Asha", "age": 16, "email": "asha-at-example.com", "pan": "12345ABCDE"},
    # Invalid: name has digits, age is not a number.
    {"name": "R2D2", "age": "twenty", "email": "r2@example.com", "pan": "ABCPK1234F"},
    # Invalid: missing fields.
    {"name": "Meena"},
]


def main() -> None:
    for index, data in enumerate(SAMPLES, start=1):
        print(f"\n--- Sample {index}: {data}")
        try:
            profile = UserProfile.model_validate(data)
            print("VALID ->", profile.model_dump())
        except ValidationError as exc:
            print(f"INVALID -> {exc.error_count()} error(s)")
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"]) or "<root>"
                print(f"  - {field}: {error['msg']}")

    print("\nJSON schema the LLM will be asked to follow in Task 3:")
    print(UserProfile.model_json_schema())


if __name__ == "__main__":
    main()
