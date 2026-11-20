// MongoDB collections and indexes (also created by the API at startup).
db = db.getSiblingDB('driftguard');
db.banking_records.createIndex({owner: 1});
db.baseline_registry.createIndex({dataset_id: 1}, {unique: true});
db.baseline_registry.createIndex({content_sha256: 1});
db.production_batches.createIndex({batch_id: 1}, {unique: true});
db.production_batches.createIndex({content_sha256: 1});
db.root_cause_reports.createIndex({job_id: 1}, {unique: true});
db.root_cause_reports.createIndex({batch_id: 1, analysed_at: -1});
db.drift_events.createIndex({job_id: 1, feature: 1});
db.monitoring_logs.createIndex({timestamp: -1});
db.stream_events.createIndex({job_id: 1, stream: 1});
db.model_evaluations.createIndex({event_id: 1}, {unique: true});
