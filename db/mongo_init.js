db = db.getSiblingDB('driftguard');
db.createCollection('prediction_logs'); db.prediction_logs.createIndex({model_id:1,batch_id:1,timestamp:-1}); db.prediction_logs.createIndex({model_id:1,record_id:1},{unique:true});
db.createCollection('production_batches'); db.production_batches.createIndex({batch_id:1},{unique:true}); db.production_batches.createIndex({model_id:1,received_at:-1});
db.createCollection('monitoring_logs'); db.monitoring_logs.createIndex({job_id:1});
db.createCollection('drift_events'); db.drift_events.createIndex({job_id:1,feature:1});
db.createCollection('root_cause_reports'); db.root_cause_reports.createIndex({job_id:1},{unique:true});
db.createCollection('experiment_runs'); db.experiment_runs.createIndex({scenario:1,seed:1});
