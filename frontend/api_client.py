"""Thin authenticated client for the DriftGuard FastAPI backend.

All statistics are computed by the backend; the UI only fetches and renders.
"""

import os
from typing import Any

import requests
import streamlit as st

API_ROOT = os.environ.get("DRIFTGUARD_API", "http://localhost:8000").rstrip("/")
BANKING = f"{API_ROOT}/api/v1/banking"


class ApiError(Exception):
    """Backend returned an error body (``detail``/``code``) or was unreachable."""

    def __init__(
        self, message: str, status: int | None = None, code: str | None = None
    ):
        super().__init__(message)
        self.status = status
        self.code = code


def _headers() -> dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _handle(resp: requests.Response) -> Any:
    if resp.status_code == 401:
        st.session_state.pop("token", None)
        raise ApiError("Session expired — please sign in again.", 401, "unauthorized")
    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("detail")
            if isinstance(detail, list):
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(str(detail), resp.status_code, body.get("code"))
        except ValueError:
            raise ApiError(f"HTTP {resp.status_code}", resp.status_code) from None
    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.content


def _request(method: str, url: str, timeout: int = 120, **kwargs) -> Any:
    try:
        resp = requests.request(
            method, url, headers=_headers(), timeout=timeout, **kwargs
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Cannot reach the DriftGuard API at {API_ROOT}. Is the backend running?",
            None,
            "unreachable",
        ) from exc
    return _handle(resp)


def bump() -> None:
    """Invalidate cached GETs after any mutation."""
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1


@st.cache_data(ttl=60, show_spinner=False)
def _cached_get(url: str, token: str, version: int, params: tuple) -> Any:
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=dict(params),
        timeout=120,
    )
    return _handle(resp)


def get(path: str, params: dict | None = None, cache: bool = True) -> Any:
    """GET a banking endpoint (cached per data version)."""
    url = f"{BANKING}{path}"
    if not cache:
        return _request("GET", url, params=params)
    try:
        return _cached_get(
            url,
            st.session_state.get("token", ""),
            st.session_state.get("data_version", 0),
            tuple(sorted((params or {}).items())),
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Cannot reach the DriftGuard API at {API_ROOT}.", None, "unreachable"
        ) from exc


def post(
    path: str, json: dict | None = None, files=None, data=None, timeout: int = 600
) -> Any:
    """POST to a banking endpoint and invalidate caches."""
    out = _request(
        "POST", f"{BANKING}{path}", json=json, files=files, data=data, timeout=timeout
    )
    bump()
    return out


def get_bytes(path: str) -> bytes:
    """GET binary content (PDF)."""
    return _request("GET", f"{BANKING}{path}", timeout=120)


def login(email: str, password: str) -> dict:
    """Authenticate and return the token payload."""
    return _request(
        "POST",
        f"{API_ROOT}/api/v1/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )


def auth_status() -> dict:
    """Whether any user account exists yet."""
    return _request("GET", f"{API_ROOT}/api/v1/auth/status", timeout=10)


def register_first_admin(name: str, email: str, password: str) -> dict:
    """Create the first administrator (only allowed while no users exist)."""
    return _request(
        "POST",
        f"{API_ROOT}/api/v1/auth/register",
        json={
            "name": name,
            "email": email,
            "password": password,
            "role": "administrator",
        },
        timeout=30,
    )


def me() -> dict:
    """Current identity."""
    return _request("GET", f"{API_ROOT}/api/v1/auth/me", timeout=30)


def health() -> dict | None:
    """Backend health, or None if unreachable."""
    try:
        return requests.get(f"{API_ROOT}/health", timeout=5).json()
    except (requests.RequestException, ValueError):
        return None
