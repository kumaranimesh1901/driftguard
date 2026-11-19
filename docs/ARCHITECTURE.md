# Architecture

DriftGuard separates ingestion, analysis, decision, storage, and presentation. A monitoring run reads model and feature metadata from PostgreSQL and prediction records from MongoDB; it writes integrity-critical summaries and alerts to PostgreSQL, and detector events, timings and root-cause report documents to MongoDB. The application joins stores only by `model_id`, `job_id`, and `batch_id`.
