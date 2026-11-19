"""ORM mirror of the prescribed relational schema."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base

JSON = JSONB


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(150), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20))


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("user_id", "name"),)
    dataset_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    name: Mapped[str] = mapped_column(String(150))
    source: Mapped[str | None] = mapped_column(String(200))
    row_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (UniqueConstraint("dataset_id", "name"),)
    feature_id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.dataset_id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(100))
    type: Mapped[str] = mapped_column(String(20))
    reference_stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("dataset_id", "algorithm", "version"),)
    model_id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.dataset_id"))
    algorithm: Mapped[str] = mapped_column(String(50))
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MonitoringJob(Base):
    __tablename__ = "monitoring_jobs"
    job_id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.model_id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    batch_id: Mapped[str] = mapped_column(String(64))
    window_start: Mapped[datetime] = mapped_column(DateTime)
    window_end: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))


class DriftSummary(Base):
    __tablename__ = "drift_summaries"
    __table_args__ = (UniqueConstraint("job_id", "feature_id"),)
    summary_id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("monitoring_jobs.job_id", ondelete="CASCADE")
    )
    feature_id: Mapped[int] = mapped_column(ForeignKey("features.feature_id"))
    psi: Mapped[float | None] = mapped_column(Float)
    ks_stat: Mapped[float | None] = mapped_column(Float)
    ks_p: Mapped[float | None] = mapped_column(Float)
    jsd: Mapped[float | None] = mapped_column(Float)
    dis: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[str | None] = mapped_column(String(10))


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("monitoring_jobs.job_id"))
    alert_type: Mapped[str] = mapped_column(String(30))
    level: Mapped[str] = mapped_column(String(10))
    recommendation: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RetrainingEvent(Base):
    __tablename__ = "retraining_events"
    event_id: Mapped[int] = mapped_column(primary_key=True)
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.alert_id"))
    old_model_id: Mapped[int] = mapped_column(ForeignKey("models.model_id"))
    new_model_id: Mapped[int] = mapped_column(ForeignKey("models.model_id"))
    decision: Mapped[str] = mapped_column(String(20))
    rollback_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
