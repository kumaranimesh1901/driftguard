"""MongoDB client and collection setup."""

from pymongo import ASCENDING, DESCENDING, MongoClient

from app.config import settings

client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=2000)


def database():
    """Return configured Mongo database."""
    return client[settings.mongo_db]


def ensure_indexes() -> None:
    """Create required MongoDB indexes idempotently."""
    db = database()
    db.prediction_logs.create_index(
        [("model_id", ASCENDING), ("batch_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    db.prediction_logs.create_index(
        [("model_id", ASCENDING), ("record_id", ASCENDING)], unique=True
    )
    db.production_batches.create_index("batch_id", unique=True)
    db.production_batches.create_index(
        [("model_id", ASCENDING), ("received_at", DESCENDING)]
    )
    db.monitoring_logs.create_index("job_id")
    db.drift_events.create_index([("job_id", ASCENDING), ("feature", ASCENDING)])
    db.root_cause_reports.create_index("job_id", unique=True)
    db.experiment_runs.create_index([("scenario", ASCENDING), ("seed", ASCENDING)])
