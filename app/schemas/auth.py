"""Authentication schemas."""

from typing import Literal

from pydantic import EmailStr, Field, field_validator

from app.schemas.common import APIModel


class RegisterRequest(APIModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: Literal["analyst", "administrator", "reviewer"] = "analyst"

    @field_validator("password")
    @classmethod
    def strong(cls, v: str) -> str:
        if not any(c.isalpha() for c in v) or not any(c.isdigit() for c in v):
            raise ValueError("password needs a letter and digit")
        return v


class LoginRequest(APIModel):
    email: EmailStr
    password: str
