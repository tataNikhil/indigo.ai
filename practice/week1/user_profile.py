"""Pydantic v2 schema for a user profile (shared by Task 2 and Task 3)."""

import re

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# PAN format: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).
# The 4th letter is the holder type (P = individual, C = company, ...).
PAN_PATTERN = re.compile(r"^[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]$")


class UserProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=100, description="Full name")
    age: int = Field(ge=18, le=120, description="Age in years (18+)")
    email: EmailStr = Field(description="Contact email address")
    pan: str = Field(description="Indian PAN number, e.g. ABCDE1234F")

    @field_validator("name")
    @classmethod
    def name_has_letters_only(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z .'-]+", value):
            raise ValueError("Name can only contain letters, spaces, '.', ''' and '-'.")
        return value.title()

    @field_validator("pan")
    @classmethod
    def pan_format(cls, value: str) -> str:
        value = value.upper()
        if not PAN_PATTERN.fullmatch(value):
            raise ValueError("PAN must look like ABCDE1234F (5 letters, 4 digits, 1 letter).")
        return value
