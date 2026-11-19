"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.auth import router as auth_router
from app.api.banking import router as banking_router
from app.core.logging_setup import configure_logging
from app.core.rate_limit import limiter
from app.database.mongo import client, ensure_indexes
from app.database.postgres import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    try:
        ensure_indexes()
    except PyMongoError:
        logger.exception("MongoDB index initialization unavailable; will retry on use")
    yield


app = FastAPI(title="DriftGuard", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.include_router(auth_router, prefix="/api/v1")
app.include_router(banking_router, prefix="/api/v1")

_DEFAULT_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
}


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Return the consistent ``{"detail": ..., "code": ...}`` error body."""
    if isinstance(exc.detail, dict) and "detail" in exc.detail:
        body = {"detail": exc.detail["detail"], "code": exc.detail.get("code")}
    else:
        body = {
            "detail": exc.detail,
            "code": _DEFAULT_CODES.get(exc.status_code, "error"),
        }
    return JSONResponse(
        status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None)
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Wrap Pydantic validation errors in the consistent error body."""
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "code": "validation_error"},
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Reject over-limit clients with 429 in the consistent error body."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "code": "rate_limited",
        },
    )


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_: Request, exc: PyMongoError) -> JSONResponse:
    """MongoDB outage: consistent 503 body, no internals leaked."""
    logger.error("MongoDB error: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Document database unavailable",
            "code": "database_unavailable",
        },
    )


@app.exception_handler(SQLAlchemyError)
async def sql_error_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    """PostgreSQL failure: consistent 503 body, no SQL leaked."""
    logger.error("PostgreSQL error: %s", type(exc).__name__)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Relational database unavailable",
            "code": "database_unavailable",
        },
    )


@app.get("/health")
def health(response: Response) -> dict:
    """Report database connectivity without leaking connection data."""
    postgres = "connected"
    mongo = "connected"
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except (OSError, SQLAlchemyError):
        postgres = "unavailable"
        logger.exception("PostgreSQL health check failed")
    try:
        client.admin.command("ping")
    except PyMongoError:
        mongo = "unavailable"
        logger.exception("MongoDB health check failed")
    if postgres == "unavailable" or mongo == "unavailable":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "postgres": postgres, "mongo": mongo}
    return {"status": "ok", "postgres": postgres, "mongo": mongo}
