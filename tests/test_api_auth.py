"""Authentication, RBAC, validation, rate limiting and error-body contract."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.rate_limit import limiter
from app.database.postgres import Base, get_db
from app.main import app

ADMIN = {
    "name": "Admin User",
    "email": "admin@example.com",
    "password": "AdminPassword123",
    "role": "administrator",
}


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)

    def override_get_db():
        db = session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    limiter.reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    limiter.reset()


def _admin_headers(client) -> dict:
    client.post("/api/v1/auth/register", json=ADMIN)
    token = client.post(
        "/api/v1/auth/login",
        json={"email": ADMIN["email"], "password": ADMIN["password"]},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _token(sub: str, minutes: int) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": sub,
            "role": "administrator",
            "iat": now,
            "exp": now + timedelta(minutes=minutes),
        },
        settings.secret_key,
        algorithm="HS256",
    )


def test_first_user_must_be_administrator(client):
    assert client.get("/api/v1/auth/status").json() == {"has_users": False}
    resp = client.post(
        "/api/v1/auth/register",
        json=ADMIN | {"role": "analyst", "email": "a@example.com"},
    )
    assert resp.status_code == 403
    assert client.post("/api/v1/auth/register", json=ADMIN).status_code == 201
    assert client.get("/api/v1/auth/status").json() == {"has_users": True}


def test_registration_after_bootstrap_needs_admin(client):
    headers = _admin_headers(client)
    other = {
        "name": "X",
        "email": "x@example.com",
        "password": "Password123",
        "role": "analyst",
    }
    assert client.post("/api/v1/auth/register", json=other).status_code == 401
    assert (
        client.post("/api/v1/auth/register", json=other, headers=headers).status_code
        == 201
    )
    assert (
        client.post("/api/v1/auth/register", json=other, headers=headers).status_code
        == 409
    )
    analyst = client.post(
        "/api/v1/auth/login",
        json={"email": other["email"], "password": other["password"]},
    ).json()
    resp = client.post(
        "/api/v1/auth/register",
        headers={"Authorization": f"Bearer {analyst['access_token']}"},
        json=other | {"email": "y@example.com"},
    )
    assert resp.status_code == 403


@pytest.mark.parametrize("token", ["not-a-valid-token", "expired", "unknown-user"])
def test_invalid_tokens_are_rejected(client, token):
    _admin_headers(client)
    value = {"expired": _token("1", -60), "unknown-user": _token("99999", 60)}.get(
        token, token
    )
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {value}"})
    assert resp.status_code == 401 and resp.json()["code"] == "unauthorized"


def test_missing_token_is_rejected(client):
    assert client.get("/api/v1/auth/me").status_code in (401, 403)


@pytest.mark.parametrize(
    "body",
    [
        {"password": "Pass1"},  # too short
        {"password": "PasswordOnly"},  # no digit
        {"is_admin": True},  # unexpected field (extra="forbid")
        {"role": "superuser"},  # unknown role
    ],
)
def test_registration_validation(client, body):
    headers = _admin_headers(client)
    payload = {
        "name": "V",
        "email": "v@example.com",
        "password": "Password123",
        "role": "analyst",
    } | body
    resp = client.post("/api/v1/auth/register", json=payload, headers=headers)
    assert resp.status_code == 422 and resp.json()["code"] == "validation_error"


def test_http_errors_use_flat_detail_code_body(client):
    client.post("/api/v1/auth/register", json=ADMIN)
    resp = client.post(
        "/api/v1/auth/login", json={"email": ADMIN["email"], "password": "wrong1234"}
    )
    assert resp.status_code == 401
    assert resp.json() == {
        "detail": "Invalid email or password",
        "code": "invalid_credentials",
    }


def test_validation_errors_carry_code(client):
    resp = client.post("/api/v1/auth/login", json={"email": "not-an-email"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error" and isinstance(
        resp.json()["detail"], list
    )


def test_login_is_rate_limited(client):
    creds = {"email": "nobody@example.com", "password": "Password123"}
    codes = [
        client.post("/api/v1/auth/login", json=creds).status_code for _ in range(7)
    ]
    assert codes[:5] == [401] * 5
    assert codes[-1] == 429
