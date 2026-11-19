"""Password and JWT security helpers."""

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import User
from app.db.postgres import get_db

hasher = PasswordHasher()
bearer = HTTPBearer()
optional_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Argon2-hash a validated password."""
    return hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify password without exposing verification errors."""
    try:
        return hasher.verify(hashed, password)
    except (InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def create_token(user: User) -> str:
    """Create a time-limited bearer token."""
    exp = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {
            "sub": str(user.user_id),
            "role": user.role,
            "iat": datetime.now(UTC),
            "exp": exp,
        },
        settings.secret_key,
        algorithm="HS256",
    )


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    """Resolve authenticated active user."""
    try:
        payload = jwt.decode(
            credentials.credentials, settings.secret_key, algorithms=["HS256"]
        )
        user = db.get(User, int(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        user = None
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"detail": "Invalid or expired token", "code": "unauthorized"},
        )
    return user


def optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
    db: Session = Depends(get_db),
) -> User | None:
    """Return a valid authenticated user when a bearer token was supplied."""
    if credentials is None:
        return None
    try:
        payload = jwt.decode(
            credentials.credentials, settings.secret_key, algorithms=["HS256"]
        )
        return db.get(User, int(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=401,
            detail={"detail": "Invalid or expired token", "code": "unauthorized"},
        )


def require_roles(*roles: str):
    """Build a role-checking FastAPI dependency."""

    def check(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"detail": "Insufficient role", "code": "forbidden"},
            )
        return user

    return check
