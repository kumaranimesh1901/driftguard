"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.auth import router as auth_router
from app.db.mongo import client, ensure_indexes
from app.db.postgres import engine
from app.logging_config import configure_logging

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
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router, prefix="/api/v1")


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
