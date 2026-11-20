# Design decisions

Rationale for choices not obvious from the code. See README.md for how the system works.

- Numeric drift and decision defaults are initial values to be calibrated experimentally, not established standards.
- PostgreSQL owns relational integrity and decisions; MongoDB owns append-heavy prediction and detector documents. Shared IDs are the integration contract.
- Demo data is synthetic (`app/services/synthetic_data.py`) so no real customer dataset is downloaded or redistributed; synthetic records are always flagged.

## Banking credit-risk platform

- **No second DIS formula.** The banking module uses the existing `dis_variants` (`DIS = D x FI x S`) and `decide` unchanged. The previous banking prototype used hard-coded FI/S weights; these were removed because they were not model-derived (SPEC section 2). FI is now mean |TreeSHAP| of the baseline XGBoost model on the production batch and S is permutation sensitivity (SPEC 11.1/11.2).
- **Noise floor.** `D` uses the bootstrap no-drift floor of SPEC 10.1 (`NULL_BOOTSTRAP_B`, `NULL_QUANTILE`), cached per (baseline, feature, batch-size bucket), instead of fixed floors. Without it a 1,000-row stable batch would show spurious small drift.
- **Numerical JSD** uses the same reference-quantile bins as PSI (shared `binned_proportions`), so the two detectors see the same partition.
- **Thresholds** come from configuration only: feature severity `SEVERITY_*`, batch level `BATCH_DIS_*`, conventional alerts `CONVENTIONAL_*` (new settings, defaults from SPEC 11.5).
- **Risk categories.** LOW/MEDIUM/HIGH follow the batch drift level. CRITICAL requires a `confirmed-degradation` alert (labels, AUC CI below baseline) with moderate/high drift, or a `severe` drop. Drift alone can never exceed HIGH. Degradation with low drift maps to HIGH (possible concept drift).
- **Baseline without labels.** No model can be trained; FI = S = 1 so DIS = D (`dis_mode = drift_only`), and model health reports that performance cannot be evaluated.
- **Model.** XGBoost in a scikit-learn pipeline (median / most-frequent imputation, one-hot employment type). The decision threshold maximises F1 on the training split; reported baseline metrics come from a 25% stratified holdout. The model is never retrained automatically.
- **Storage.** PostgreSQL: `datasets`, `features` (reference profile in `reference_stats`), `models`, `monitoring_jobs`, `drift_summaries`, `alerts`. MongoDB: `banking_records` (one document per customer row, avoiding the 16 MB document limit), `baseline_registry`, `production_batches`, `root_cause_reports`, `drift_events`, `monitoring_logs` (audit). `drift_summaries` columns are REAL, so values are clamped to the float4 range there (e.g. KS p = 1e-300 -> 0); full precision stays in the Mongo report.
- **Duplicates.** Uploading an existing batch ID, or byte-identical content under another ID, is rejected unless `overwrite` / `allow_duplicate_content` is set. Re-analysing identical inputs (same baseline and batch content hashes) returns the stored report unless `force` is set.
- **Synthetic scenario B** was calibrated to +22% income / +15% credit limit so it lands in the MEDIUM band at demo scale; the earlier +30% shift gave income PSI ~0.4 (conventionally "significant") and straddled the HIGH boundary. A labelled `concept_shift` option changes the 2026 outcome relationship so confirmed degradation (CRITICAL) can be demonstrated.

## Project simplification (2026-11-20)

- **One user-facing application**: the Streamlit dashboard (`frontend/dashboard.py`). FastAPI, PostgreSQL, MongoDB and all statistics run in the background; `./start.sh` / `./stop.sh` manage them.
- Docker Compose runs only the databases; the API and dashboard run on the host from `.venv`, so there is a single way to start the project.
- The original generic loan generator (`app/simulation`) was removed: nothing in the application used it after the banking generator replaced it.
- Print-only audit scripts were converted into pytest assertions (`tests/test_drift_methods.py`, `tests/test_api_auth.py`); probes for endpoints that were specified but never built were dropped.
- The audit trail is still written for every upload, analysis and report; it is shown inside Batch Monitoring rather than as a separate page.

## Audit corrections and additions (2026-11-20)

- **Noise-floor null design fixed** (`calibration._draw_pair`). The bootstrap previously compared two *batch-sized* samples, but the real comparison is the full reference (N) against one batch (n). The null now draws N and n with replacement from the reference. Measured false-alarm rate at the 95% floor on genuinely stable data: before 0% (PSI) / 1.25% (KS), after 3.25% / 4.75% (nominal 5%). Effect: D was biased downwards; it is now calibrated.
- **Detector agreement** is reported per feature (`detector_agreement`: `agree-drift`, `agree-stable`, `disagree`, plus which detectors exceed their floor and whether KS is significant). D still aggregates with `max`, but disagreement is visible instead of hidden.
- **Low-impact strong drift** no longer reads as "distributions remain stable". A feature with high D but DIS below `BATCH_DIS_MODERATE` is listed in `low_impact_drift_features` with a data-quality/pipeline check action. DIS still decides model risk.
- **Drift types kept distinct** (`drift_types`): data drift (any D severity), prediction drift (ADWIN on predicted probabilities, or KS on outputs), suspected concept drift (labels: degradation or error-stream change with low DIS), confirmed degradation (labels: AUC CI). Without labels the last two are `null` (unknown), never `false`.
- **ADWIN** (`app/services/drift/adwin.py`, River) runs on two defined streams: predicted probability (prediction drift, no labels) and |y − p| (error level, labels). Reference segment = the champion's holdout signal (saved as `<model>.joblib.holdout.npz`; legacy artifacts re-create the identical split). Results are in the report and in MongoDB `stream_events`. ADWIN is not used for feature-distribution drift.
- **Champion–challenger retraining** (`app/services/monitoring/retraining.py`, `/banking/models/*`). Labelled data only; 40% of labelled production rows form an independent post-drift test set never used for fitting or threshold choice; the baseline holdout is kept as a no-regression guard. Deploy requires AUC gain ≥ `RETRAIN_MIN_GAIN_AUC`, paired-bootstrap 95% CI > 0, and no baseline regression > `PERF_SEVERE_DROP`; it is an administrator action in one transaction (`retraining_events`, `models.status`). Reject and rollback are supported. Evaluations are stored in MongoDB `model_evaluations`. Analyses use the current champion; re-analysing a batch the champion was trained on sets `leakage_warning`.
- **Drift reference stays the registered baseline** after a deployment; only the model (FI, S, performance reference AUC, ADWIN reference) changes. Re-baselining is a separate, explicit action (register a new baseline).
- **Analysis reuse signature** now includes the champion model id and a fingerprint of all result-affecting settings, so changing a threshold or deploying a model never returns a stale stored report.
- **Partial-write compensation** in `complete_job` now covers both Mongo writes (report and events); previously a failed `drift_events` insert left an orphan report for a failed job.
- **CORS** no longer allows `*` together with credentials.
- **Research experiment**: `scripts/experiments/dis_prioritisation.py` compares DIS and its ablations against conventional PSI/KS ranking using a counterfactual ground truth (undo one feature's drift by quantile mapping onto the reference and measure the change in model outputs). Results are written to `docs/experiments/`.
