"""MongoDB client and index setup.

Collections:
    banking_records      one document per customer record (baseline or batch)
    baseline_registry    registered baseline metadata
    production_batches   batch metadata and processing history
    root_cause_reports   full analysis report per monitoring job
    drift_events         per-feature detector values per job
    stream_events        ADWIN prediction / error stream results per job
    model_evaluations    champion-vs-challenger evaluation records (retraining)
    monitoring_logs      audit trail
"""

from pymongo import ASCENDING, DESCENDING, MongoClient

from app.core.config import settings

client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=2000)


def database():
    """Return the configured Mongo database."""
    return client[settings.mongo_db]


def ensure_indexes() -> None:
    """Create the indexes used by the application (idempotent)."""
    db = database()
    db.banking_records.create_index([("owner", ASCENDING)])
    db.baseline_registry.create_index("dataset_id", unique=True)
    db.baseline_registry.create_index("content_sha256")
    db.production_batches.create_index("batch_id", unique=True)
    db.production_batches.create_index("content_sha256")
    db.root_cause_reports.create_index("job_id", unique=True)
    db.root_cause_reports.create_index(
        [("batch_id", ASCENDING), ("analysed_at", DESCENDING)]
    )
    db.drift_events.create_index([("job_id", ASCENDING), ("feature", ASCENDING)])
    db.stream_events.create_index([("job_id", ASCENDING), ("stream", ASCENDING)])
    db.model_evaluations.create_index("event_id", unique=True)
    db.monitoring_logs.create_index([("timestamp", DESCENDING)])
