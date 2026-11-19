"""Request schemas for the banking monitoring API."""

from datetime import date
from typing import Literal

from pydantic import Field, field_validator

from app.schemas.common import APIModel
from app.services.datasets import BATCH_ID_PATTERN


def _check_batch_id(value: str) -> str:
    if not BATCH_ID_PATTERN.match(value):
        raise ValueError(
            "batch_id must start with a letter or digit and contain only letters, "
            "digits, '_' or '-' (max 64 characters)"
        )
    return value


class AnalysisRequest(APIModel):
    batch_ids: list[str] = Field(min_length=1, max_length=50)
    mode: Literal["individual", "combined"] = "individual"
    force: bool = False

    @field_validator("batch_ids")
    @classmethod
    def valid_ids(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("batch_ids must be unique")
        return [_check_batch_id(x) for x in v]


class RunAllRequest(APIModel):
    force: bool = False


class GenerateBatchRequest(APIModel):
    scenario: Literal["A", "B", "C", "D"]
    batch_id: str
    batch_date: date
    n: int = Field(default=1000, ge=100, le=20000)
    description: str = Field(default="", max_length=300)
    with_labels: bool = False
    concept_shift: bool = False
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    overwrite: bool = False

    @field_validator("batch_id")
    @classmethod
    def valid_id(cls, v: str) -> str:
        return _check_batch_id(v)


class GenerateBaselineRequest(APIModel):
    n: int = Field(default=2500, ge=200, le=50000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    dataset_name: str = Field(
        default="2020_Synthetic_Baseline", min_length=1, max_length=120
    )


class SeedDemoRequest(APIModel):
    include_concept_shift_batch: bool = True
    force: bool = False
