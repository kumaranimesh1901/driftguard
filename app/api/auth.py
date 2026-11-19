"""Auth endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.rate_limit import LOGIN_RATE_LIMIT, limiter
from app.core.security import (
    create_token,
    current_user,
    hash_password,
    optional_current_user,
    verify_password,
)
from app.database.models import User
from app.database.postgres import get_db
from app.schemas.auth import LoginRequest, RegisterRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
def auth_status(db: Session = Depends(get_db)) -> dict:
    """Whether any user exists (the dashboard offers first-admin setup if not)."""
    return {"has_users": bool(db.scalar(func.count(User.user_id)))}


@router.post("/register", status_code=201)
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
    actor: User | None = Depends(optional_current_user),
) -> dict:
    """Create a user; bootstrap only allows the first administrator."""
    count = db.scalar(func.count(User.user_id)) or 0
    if count and actor is None:
        raise HTTPException(
            401,
            detail={
                "detail": "Authentication is required to create users",
                "code": "unauthorized",
            },
        )
    if count and actor.role != "administrator":
        raise HTTPException(
            403, detail={"detail": "Administrator role required", "code": "forbidden"}
        )
    if not count and body.role != "administrator":
        raise HTTPException(
            403,
            detail={"detail": "First user must be administrator", "code": "forbidden"},
        )
    if db.query(User).filter_by(email=str(body.email)).first():
        raise HTTPException(
            409, detail={"detail": "Email already exists", "code": "conflict"}
        )
    user = User(
        name=body.name,
        email=str(body.email),
        password_hash=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "user_id": user.user_id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }


@router.post("/login")
@limiter.limit(LOGIN_RATE_LIMIT)
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)) -> dict:
    """Authenticate and return a JWT."""
    user = db.query(User).filter_by(email=str(body.email)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            401,
            detail={
                "detail": "Invalid email or password",
                "code": "invalid_credentials",
            },
        )
    return {
        "access_token": create_token(user),
        "token_type": "bearer",
        "role": user.role,
        "name": user.name,
    }


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    """Return the bearer-token identity."""
    return {
        "user_id": user.user_id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }
