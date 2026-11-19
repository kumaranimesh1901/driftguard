# DRIFTGUARD: COMPLETE BUILD SPECIFICATION FOR A CODING AGENT

You are an expert full-stack and machine learning engineer. Build the complete, runnable project described below from scratch, in a single repository. Read this entire document before writing code. Work in the phases of Section 25, produce an implementation plan first, and verify each phase (run the code, run the tests) before moving to the next.

---

## 0. HOW TO WORK

1. First produce a short implementation plan (files, order, risks). Then build phase by phase (Section 25).
2. After each phase, run the tests and a smoke check. Fix failures before continuing. Do not leave stubs or `TODO` placeholders in delivered code. If something must be left out, list it in `docs/KNOWN_LIMITATIONS.md`.
3. Where this document gives a formula, threshold or default, implement it as specified and make it configurable. Where it is silent, choose the simplest correct option and record the decision in `docs/DECISIONS.md`.
4. If a library version conflict appears (for example SHAP, NumPy, XGBoost), resolve it by pinning compatible versions in `requirements.txt` and note it in `docs/DECISIONS.md`.
5. At the end, run the full verification checklist in Section 26 and report the actual outputs.

## 1. PROJECT SUMMARY AND CONTEXT

**Name:** DriftGuard: An Intelligent Framework for Automated Data Drift Detection, Root-Cause Analysis, and ML Model Monitoring.

**Context:** This is a college project for a Database Management Systems Lab (course CD252IA, Dept. of CSE, RV College of Engineering, Bengaluru). It is graded on database design (ER model, relational schema, normalisation, SQL and NoSQL queries, hybrid SQL+NoSQL integration), data flow design, forms/security/validation, a working demo with a neat UI, and a sound research idea. So the database layer and the UI must be of high quality, not an afterthought.

**What the system does:** It monitors a deployed machine learning model (a loan-default classifier) for data drift. For each incoming production batch it:
1. detects drift per feature using PSI, KS test, Jensen–Shannon divergence and ADWIN;
2. normalises these measures against a no-drift bootstrap reference;
3. computes SHAP-based feature importance and a permutation-style model sensitivity per feature;
4. computes a **Drift Impact Score (DIS)** per feature and ranks features by it;
5. monitors model performance when true labels are available (labels can be delayed);
6. decides an action (continue, investigate, consider retraining, immediate review) with a documented rule engine;
7. stores metadata and decisions in **PostgreSQL**, and large/flexible documents (prediction logs, drift events, reports, logs) in **MongoDB**;
8. shows everything in a dashboard;
9. supports controlled retraining with a champion–challenger comparison, admin approval and rollback.

It also contains a **drift simulation module** and an **experiment runner** that compare DriftGuard with conventional drift alerts.

**Research question (the experiments must be able to test it):** *Can combining statistical drift magnitude with model-specific feature importance and observed model-performance changes improve the prioritisation of potentially consequential drift compared with conventional drift alerts?*

## 2. NON-NEGOTIABLE RULES

- **No fabricated results.** Never hard-code or invent accuracy, AUC, detection or ranking numbers anywhere (code, README, docs, UI). Every number shown must come from a real run. Dashboard values come from the databases. The experiment runner writes real outputs to files.
- **DIS is a proposed research component, not an established standard.** In docs and UI, call it "proposed". Do not claim DriftGuard is the first system for drift detection or monitoring.
- **Drift is not degradation.** The UI and API must always distinguish a **drift-warning** (no labels yet) from **confirmed-degradation** (labels show a real performance drop).
- **PSI, KS and JSD are different measures on different scales.** Never average raw values. Only combine them after the normalisation of Section 10, and keep per-detector values visible.
- **Security basics are mandatory** (Section 19): hashed passwords, JWT auth, role checks on every endpoint, parameterised SQL only, validated inputs, no secrets in code.
- **Keep the eight core PostgreSQL tables exactly as named and shaped in Section 6.1**, because the course documents (ER diagram, normalisation) refer to them. Two supporting tables (`thresholds`, `threshold_history`) are allowed and listed separately.
- Python 3.11, type hints on all public functions, docstrings, logging (no bare `print` in library code), deterministic random seeds from configuration.

## 3. TECHNOLOGY STACK

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Backend | FastAPI, Uvicorn, Pydantic v2, pydantic-settings |
| Frontend | Streamlit (multi-page), Plotly for charts, `requests` to call the API |
| ML | scikit-learn, XGBoost, SHAP, joblib; models: RandomForest, XGBoost, LogisticRegression |
| Data | pandas, NumPy, SciPy, pyarrow (parquet) |
| Drift | Own implementations of PSI, KS (via `scipy.stats.ks_2samp`), JSD (via `scipy.spatial.distance.jensenshannon`); ADWIN from the `river` package. Optional cross-check script using Evidently or Frouros, not a runtime dependency |
| SQL DB | PostgreSQL 15, SQLAlchemy 2.x (ORM + Core), psycopg2-binary |
| NoSQL DB | MongoDB 6, PyMongo |
| Auth | `argon2-cffi` for password hashing, `PyJWT` for tokens |
| Scheduler | APScheduler (optional automation of monitoring jobs) |
| Testing | pytest, httpx (FastAPI TestClient), `mongomock` or a test Mongo container, pytest-cov |
| Tooling | Docker, docker-compose, Makefile, `.env.example`, Git |

Provide `requirements.txt` (runtime) and `requirements-dev.txt` (tests, linting with ruff and black).

## 4. REPOSITORY STRUCTURE

Create exactly this layout (add files if needed, do not drop any):

```
driftguard/
  README.md
  Makefile
  docker-compose.yml
  .env.example
  .gitignore
  requirements.txt
  requirements-dev.txt
  docs/
    ARCHITECTURE.md        # layers, data flow, how SQL and NoSQL work together
    DECISIONS.md           # every choice not fixed by this spec
    KNOWN_LIMITATIONS.md
    API.md                 # generated or hand-written endpoint reference
  db/
    init.sql               # PostgreSQL DDL from Section 6.1 (+ supporting tables)
    mongo_init.js          # collections and indexes from Section 6.2
  queries/
    sql_queries.sql        # S1..S12 (Section 23)
    mongo_queries.js       # N1..N9 (Section 23)
  app/
    main.py                # FastAPI app factory, routers, CORS, startup
    config.py              # pydantic-settings
    logging_config.py
    security.py            # hashing, JWT, role dependencies
    db/
      postgres.py          # engine, session
      mongo.py             # client, collections
      models.py            # SQLAlchemy ORM for the tables
    schemas/               # Pydantic request/response models
    api/
      auth.py datasets.py models.py batches.py labels.py monitoring.py
      alerts.py retraining.py thresholds.py simulation.py experiments.py
    services/
      preprocessing.py     # ColumnTransformer builder, data-quality report
      training.py          # train baseline, evaluate, save artifacts
      reference_profile.py # bins, proportions, null calibration, global FI
      drift/
        psi.py ks.py jsd.py adwin_monitor.py
        normalise.py       # null-based normalisation (Section 10)
        conventional.py    # baseline conventional alerts (Section 11.5)
        feature_drift.py   # runs all detectors for one batch
      explain/
        shap_importance.py sensitivity.py dis.py
      performance.py       # metrics, bootstrap CI, degradation state
      decision.py          # rule engine (Section 13)
      pipeline.py          # run_monitoring_job orchestration (Section 14)
      retraining.py        # champion-challenger, deploy, rollback
      scheduler.py
    simulation/
      generator.py         # synthetic loan data generator
      scenarios.py         # drift scenarios and patterns
      kaggle_adapter.py    # optional adapter for a real CSV
    experiments/
      runner.py            # multi-seed experiments
      metrics.py           # evaluation metrics
      plots.py
  frontend/
    Home.py                # Streamlit entry (login)
    pages/
      1_Datasets.py 2_Models.py 3_Monitoring.py 4_Drift_Results.py
      5_Performance.py 6_Alerts_and_Retraining.py 7_Simulation_Lab.py
      8_Experiments.py 9_Admin.py
    api_client.py
    ui_helpers.py
  scripts/
    seed_demo.py           # creates users, dataset, model, batches, alerts
    run_demo.py            # scripted 8-step demo against the API
    run_queries.py         # runs queries/ files and prints results
  artifacts/               # runtime: models, reference profiles, adwin state (gitignored)
  data/                    # runtime: raw and generated data (gitignored)
  experiments/results/     # runtime: CSV, plots (gitignored except .gitkeep)
  tests/
    unit/ integration/
```

## 5. CONFIGURATION

All settings come from environment variables via `app/config.py` (pydantic-settings), with `.env.example` documenting them:

```
APP_ENV=dev
SECRET_KEY=change-me                 # JWT signing key (never commit a real one)
ACCESS_TOKEN_EXPIRE_MINUTES=60
POSTGRES_URL=postgresql+psycopg2://driftguard:driftguard@postgres:5432/driftguard
MONGO_URL=mongodb://mongo:27017
MONGO_DB=driftguard
ARTIFACT_DIR=./artifacts
RANDOM_SEED=42
MAX_UPLOAD_MB=50
MAX_BATCH_ROWS=50000
# drift and decision defaults (all editable by admin through the API, see Section 18.9)
PSI_BINS=10
NULL_BOOTSTRAP_B=200
NULL_QUANTILE=0.95
NORM_CAP_PSI=0.5
NORM_CAP_KS=0.5
NORM_CAP_JSD=0.3
D_AGGREGATION=max                    # max | mean
SEVERITY_MODERATE=0.15               # on normalised D
SEVERITY_HIGH=0.50
BATCH_DIS_MODERATE=0.05
BATCH_DIS_HIGH=0.25
PERF_MEANINGFUL_DROP=0.02            # absolute AUC drop
PERF_SEVERE_DROP=0.05
PERF_BOOTSTRAP_B=200
RETRAIN_MIN_GAIN_AUC=0.01
SENSITIVITY_REPEATS=5
SHAP_SAMPLE_SIZE=2000
```

State clearly in `docs/DECISIONS.md` that the numeric defaults are **initial values to be calibrated experimentally**, not established standards.

## 6. DATABASE DESIGN

### 6.1 PostgreSQL (relational, integrity-critical data)

Implement in `db/init.sql` and mirror in SQLAlchemy models. Names and columns of the eight core tables must match exactly:

```sql
CREATE TABLE users (
  user_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(150) NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role VARCHAR(20) NOT NULL CHECK (role IN ('analyst','administrator','reviewer'))
);

CREATE TABLE datasets (
  dataset_id SERIAL PRIMARY KEY,
  user_id INT NOT NULL REFERENCES users(user_id),
  name VARCHAR(150) NOT NULL,
  source VARCHAR(200),
  row_count INT CHECK (row_count >= 0),
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE (user_id, name)
);

CREATE TABLE features (
  feature_id SERIAL PRIMARY KEY,
  dataset_id INT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  name VARCHAR(100) NOT NULL,
  type VARCHAR(20) NOT NULL CHECK (type IN ('numerical','categorical')),
  reference_stats JSONB,
  UNIQUE (dataset_id, name)
);

CREATE TABLE models (
  model_id SERIAL PRIMARY KEY,
  dataset_id INT NOT NULL REFERENCES datasets(dataset_id),
  algorithm VARCHAR(50) NOT NULL,
  version INT NOT NULL,
  status VARCHAR(20) NOT NULL CHECK (status IN ('champion','challenger','retired')),
  metrics JSONB,
  trained_at TIMESTAMP DEFAULT now(),
  UNIQUE (dataset_id, algorithm, version)
);

CREATE TABLE monitoring_jobs (
  job_id SERIAL PRIMARY KEY,
  model_id INT NOT NULL REFERENCES models(model_id),
  user_id INT NOT NULL REFERENCES users(user_id),
  batch_id VARCHAR(64) NOT NULL,
  window_start TIMESTAMP NOT NULL,
  window_end TIMESTAMP NOT NULL,
  status VARCHAR(20) NOT NULL,          -- queued | running | completed | failed
  CHECK (window_end >= window_start)
);

CREATE TABLE drift_summaries (
  summary_id SERIAL PRIMARY KEY,
  job_id INT NOT NULL REFERENCES monitoring_jobs(job_id) ON DELETE CASCADE,
  feature_id INT NOT NULL REFERENCES features(feature_id),
  psi REAL, ks_stat REAL, ks_p REAL, jsd REAL, dis REAL,
  severity VARCHAR(10) CHECK (severity IN ('low','moderate','high')),
  UNIQUE (job_id, feature_id)
);

CREATE TABLE alerts (
  alert_id SERIAL PRIMARY KEY,
  job_id INT NOT NULL REFERENCES monitoring_jobs(job_id),
  alert_type VARCHAR(30) NOT NULL CHECK (alert_type IN ('drift-warning','confirmed-degradation')),
  level VARCHAR(10) NOT NULL,
  recommendation TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE retraining_events (
  event_id SERIAL PRIMARY KEY,
  alert_id INT REFERENCES alerts(alert_id),            -- NULL for manual retraining
  old_model_id INT NOT NULL REFERENCES models(model_id),
  new_model_id INT NOT NULL REFERENCES models(model_id),
  decision VARCHAR(20) NOT NULL CHECK (decision IN ('pending','deployed','rejected','rolled_back')),
  rollback_flag BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT now(),
  CHECK (old_model_id <> new_model_id)
);
```

Notes: add `'pending'` to the decision list as shown (a challenger waits for admin approval). Add indexes: `drift_summaries(job_id)`, `monitoring_jobs(model_id, window_start)`, `alerts(job_id, created_at)`.

**Supporting tables (allowed extras):**

```sql
CREATE TABLE thresholds (
  threshold_id SERIAL PRIMARY KEY,
  name VARCHAR(60) NOT NULL UNIQUE,     -- e.g. 'severity_moderate'
  value DOUBLE PRECISION NOT NULL,
  updated_by INT REFERENCES users(user_id),
  updated_at TIMESTAMP DEFAULT now()
);
CREATE TABLE threshold_history (
  history_id SERIAL PRIMARY KEY,
  name VARCHAR(60) NOT NULL,
  old_value DOUBLE PRECISION, new_value DOUBLE PRECISION NOT NULL,
  changed_by INT REFERENCES users(user_id),
  changed_at TIMESTAMP DEFAULT now()
);
```

Seed `thresholds` from the Section 5 defaults. The pipeline reads thresholds from this table (falling back to config).

**Transactions:** deploying a challenger (champion → retired, challenger → champion, insert/update `retraining_events`) must be one transaction. Guarantee at most one `champion` per dataset (partial unique index: `CREATE UNIQUE INDEX one_champion_per_dataset ON models(dataset_id) WHERE status='champion';`).

### 6.2 MongoDB (large, append-heavy, flexible documents)

Database `driftguard`. Create collections and indexes in `db/mongo_init.js` and also ensure them at app startup:

| Collection | Document shape | Indexes |
|---|---|---|
| `prediction_logs` | `{record_id, model_id, batch_id, timestamp, features:{...}, predicted_prob, predicted_label, true_label (null until labels arrive)}` | `{model_id:1, batch_id:1, timestamp:-1}`, unique `{model_id:1, record_id:1}` |
| `production_batches` | `{batch_id, model_id, received_at, n_rows, schema_ok, uploaded_by, summary_stats:{feature:{mean,std,...}}}` | `{batch_id:1}` unique, `{model_id:1, received_at:-1}` |
| `monitoring_logs` | `{job_id, model_id, started_at, finished_at, durations:{psi_ms, ks_ms, jsd_ms, adwin_ms, shap_ms, sens_ms}, status, error}` | `{job_id:1}` |
| `drift_events` | `{job_id, model_id, feature, detector, value, noise_floor_q95, normalised, exceeded_conventional, histogram:{ref:[...], prod:[...], edges:[...]}, adwin:{detected, index}}` | `{job_id:1, feature:1}` |
| `root_cause_reports` | `{job_id, model_id, created_at, batch_drift_level, performance_state, ranked_features:[{feature, D, FI, S, dis, severity, detectors_flagged, dis_variants:{d_only, d_fi, weighted_sum, rank_based}}], recommendation, notes}` | `{job_id:1}` unique |
| `experiment_runs` | `{run_id, scenario, seed, params, results:{...}, created_at}` | `{scenario:1, seed:1}` |

Link rule: PostgreSQL and MongoDB are joined **in the application by shared keys** (`model_id`, `job_id`, `batch_id`). No data is duplicated across stores except the keys. If a MongoDB write fails after the PostgreSQL job row exists, set job status to `failed` and log the error.

## 7. DATA

### 7.1 Primary data source: built-in synthetic loan generator (no download needed)

Implement `app/simulation/generator.py` with a class `LoanDataGenerator(seed)` that produces realistic loan-default data with ground-truth structure, so drift scenarios are controllable.

Features:
- Numerical: `age` (int, 21–75), `income` (annual, log-normal, median about 45,000), `loan_amount` (log-normal, related to income), `credit_score` (normal clipped 300–850, mean about 680, sd about 70), `debt_to_income` (0–1, beta distribution), `employment_years` (0–40), `num_credit_lines` (int 0–15), `interest_rate` (derived from credit score plus noise).
- Categorical: `employment_type` (salaried, self-employed, contract, unemployed) with default proportions about 0.62, 0.20, 0.14, 0.04; `home_ownership` (rent, own, mortgage); `loan_purpose` (education, home, medical, business, personal).
- Target `default` (0/1): sampled from a logistic model of the features, with interpretable coefficients (lower credit score, higher debt-to-income, higher loan-to-income, unemployed or self-employed raise risk; higher income and longer employment lower it). Target default rate about 15–20 percent. Store the coefficient vector in the class so a **concept-drift scenario** can change it.
- Make several features deliberately **low importance** (for example `num_credit_lines`, `home_ownership`) so that "drift in a low-importance feature" is a real control scenario.

Provide `generate(n, drift=None, batch_index=0)` where `drift` is a `DriftSpec` (Section 16) applied to the feature distributions and/or the label function.

### 7.2 Optional: real CSV adapter

`app/simulation/kaggle_adapter.py` lets the user place a public credit-risk CSV in `data/raw/` and map its columns to the feature schema with a YAML mapping file (`data/mapping.example.yaml`). The system must work fully without it. Do not download anything automatically. Document that the dataset licence must permit academic use.

## 8. ML CORE

**Preprocessing** (`preprocessing.py`): `ColumnTransformer` with median imputation + `StandardScaler` for numerical features (scaling optional for tree models, required for logistic regression) and most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')` for categorical features. Keep a mapping from encoded columns back to original feature names (needed for SHAP aggregation). Also produce a data-quality report: missing-value counts, duplicate rows, constant columns, type inference, out-of-range values; return as JSON.

**Training** (`training.py`): stratified split of the dataset into train 60 percent, reference 20 percent, test 20 percent (seeded). Algorithms: `RandomForest`, `XGBoost`, `LogisticRegression` (choice from the API). Evaluate on the test split: accuracy, precision, recall, F1, ROC-AUC, confusion matrix. Save with joblib: `artifacts/models/{model_id}.joblib` (pipeline = preprocessor + classifier). Store metrics in `models.metrics` (JSONB) including `baseline_auc` (the AUC used as the performance baseline for degradation checks).

**Reference profile** (`reference_profile.py`): built from the **reference split** (not the training data, to avoid optimistic bias) and saved at `artifacts/reference/{model_id}.json` plus a sample parquet `artifacts/reference/{model_id}_sample.parquet` (up to 20,000 rows). Content:
- Per numerical feature: quantile-based bin edges (10 bins, unique edges, with open-ended first and last bin), reference proportions per bin, basic stats (mean, std, quantiles).
- Per categorical feature: category list and proportions, plus an `__other__` bucket for unseen categories.
- Global SHAP feature importance (Section 11.1) computed on the reference sample.
- Null calibration quantiles per feature and detector (Section 10.1).
- Copy the numerical/categorical metadata into `features.reference_stats` (JSONB).

## 9. DRIFT DETECTION ALGORITHMS

All detectors compare the **reference sample** (the stored reference data) with the **production batch**, per feature.

### 9.1 PSI (`psi.py`)
`PSI = sum_b (p_b - q_b) * ln(p_b / q_b)` where `p_b` and `q_b` are the production and reference proportions in bin `b`. Use the reference bin edges. Clip proportions to a minimum of `1e-4` before the log. For categorical features, bins are the categories plus `__other__`.

### 9.2 KS test (`ks.py`)
Numerical features only. `scipy.stats.ks_2samp(ref, prod)`; return the statistic `D` and the p-value. Always report the statistic (effect size) together with the p-value because large samples make tiny shifts significant. For categorical features store `NULL` for KS in `drift_summaries`.

### 9.3 JSD (`jsd.py`)
Build probability vectors from the same bins/categories as PSI (with small smoothing). `JSD = scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2` (square the returned distance to get the divergence, which lies in [0,1]).

### 9.4 ADWIN (`adwin_monitor.py`)
Use `river.drift.ADWIN`. Keep one ADWIN instance per `(model_id, numerical feature)`, fed with the feature values of the incoming records in timestamp order, and one for the **error stream** (`error = 1 if predicted_label != true_label`) when labels exist. Persist instance state between batches (pickle to `artifacts/adwin/{model_id}_{feature}.pkl`). For each batch report `{detected: bool, index: first change position or null}`. Explain in the docs that ADWIN works on streams and gives change-time information, whereas PSI/KS/JSD compare a batch with the reference.

### 9.5 Why these are kept separate
Each detector has its own scale and sensitivity. Never combine raw values. Show all values per feature in the UI. Record **detector agreement**: the list of detectors whose value exceeds their own noise floor (Section 10.1). When detectors disagree, show that explicitly as an informative signal.

## 10. CALIBRATION AND NORMALISATION

### 10.1 Bootstrap null (noise floor)
For each feature and each of PSI, KS, JSD, estimate the distribution of the statistic **when there is no drift**: draw `B=200` pairs of samples (each of size `n_batch`, drawn with replacement from the reference sample, one as "reference-like", one as "batch-like", using disjoint halves of the reference sample where possible), compute the statistic for each pair, and store `q50`, `q95`, `q99`. Because the null depends on batch size, compute it for the actual `n_batch` on demand and cache by `(model_id, feature, n_batch_bucket)`. Define the **noise floor** as the `NULL_QUANTILE` (default `q95`).

### 10.2 Normalised drift magnitude
For detector `m` and feature `i`, with observed value `s` and noise floor `f` and cap `c` (config: PSI 0.5, KS 0.5, JSD 0.3):

`Dn_m,i = clip( (s - f) / (c - f), 0, 1 )`

Aggregate to one number per feature: `D_i = max_m Dn_m,i` (default) or `mean_m Dn_m,i` (configurable through `D_AGGREGATION`). Implement both so the ablation can compare them. Categorical features use PSI and JSD only.

### 10.3 Per-feature severity
`low` if `D_i < SEVERITY_MODERATE`, `moderate` if below `SEVERITY_HIGH`, else `high` (defaults 0.15 and 0.50). Store in `drift_summaries.severity`.

## 11. EXPLAINABILITY, SENSITIVITY AND THE DRIFT IMPACT SCORE

### 11.1 Feature importance FI (`shap_importance.py`)
- Compute SHAP values on the **transformed** feature matrix of a random sample (`SHAP_SAMPLE_SIZE`, default 2000) of the current production batch (and once on the reference sample for the stored global importance). Use `shap.TreeExplainer` for RandomForest/XGBoost and `shap.LinearExplainer` for logistic regression.
- Handle SHAP version differences: for binary classifiers some versions return a list of two arrays or an array with a trailing class dimension. Always take the values for the positive class.
- Importance of an encoded column = mean absolute SHAP value. **Aggregate one-hot columns back to the original categorical feature by summing their mean absolute SHAP values.**
- Scale to `FI_i = importance_i / max_j importance_j`, giving a value in [0,1].

### 11.2 Sensitivity S (`sensitivity.py`)
For feature `i` and the production batch `X`: repeat `SENSITIVITY_REPEATS` times: copy `X`, **randomly permute only column `i` within the batch** (this resamples the feature from the production distribution and breaks its link to the other features), predict probabilities, and compute the mean absolute change versus the original predicted probabilities. Average over repeats to get `S_raw_i`, then `S_i = S_raw_i / max_j S_raw_j`. Document that this is permutation-style sensitivity of the model output to the feature.

### 11.3 Drift Impact Score (`dis.py`), the proposed component
`DIS_i = D_i * FI_i * S_i`, each term in [0,1]. Store `dis` in `drift_summaries`. Also compute and store (in `root_cause_reports.ranked_features[].dis_variants`) these ablation variants:
1. `d_only = D_i`
2. `d_fi = D_i * FI_i`
3. `weighted_sum = 0.5*D_i + 0.25*FI_i + 0.25*S_i`
4. `rank_based = mean of the per-feature ranks of D, FI and S` (lower rank number is more important; convert to a [0,1] score)

Note in docs that a pure product is zero whenever any factor is zero, which is why the variants exist, and that DIS validity must be tested by experiments, not assumed.

### 11.4 Batch-level drift level
`batch_drift_level`: `high` if `max_i DIS_i >= BATCH_DIS_HIGH`, `moderate` if `>= BATCH_DIS_MODERATE`, else `low` (defaults 0.25 and 0.05; to be calibrated in experiments).

### 11.5 Conventional baseline alerts (`conventional.py`)
For comparison, implement conventional per-feature alerts that ignore the model: a feature alerts if **PSI > 0.25** or **KS p-value < 0.01** or **JSD > 0.1** (all configurable) or ADWIN detected change. A batch-level conventional alert fires if any feature alerts. Store these flags in `drift_events.exceeded_conventional`.

## 12. PERFORMANCE MONITORING (`performance.py`)

- When labels exist for at least `MIN_LABELS` (default 200) records of a batch, compute accuracy, precision, recall, F1, ROC-AUC for the batch. Compute a bootstrap confidence interval for AUC (`PERF_BOOTSTRAP_B` resamples).
- Compare with `baseline_auc` in `models.metrics`.
- `performance_state`:
  - `unknown`: not enough labels.
  - `stable`: upper CI bound of batch AUC >= baseline_auc - `PERF_MEANINGFUL_DROP`.
  - `degraded`: upper CI bound < baseline_auc - `PERF_MEANINGFUL_DROP`.
  - `severe`: upper CI bound < baseline_auc - `PERF_SEVERE_DROP`.
- Labels arrive later through `POST /labels`; the job for a batch can be **re-evaluated** (`POST /monitoring/jobs/{job_id}/reevaluate`) once labels are present, which updates the performance state, may create a new `confirmed-degradation` alert and updates the root-cause report.

## 13. DECISION ENGINE (`decision.py`)

Inputs: `batch_drift_level` (low/moderate/high), `performance_state` (unknown/stable/degraded/severe). Output: `alert_type`, `level`, `recommendation`.

| Drift level | Performance | Alert type | Recommendation |
|---|---|---|---|
| low | unknown or stable | none | Continue monitoring |
| moderate | unknown | drift-warning | Investigate and monitor closely; labels needed to confirm impact |
| high | unknown | drift-warning | Investigate affected features and data quality; labels needed to confirm impact |
| moderate | stable | drift-warning | Investigate and monitor closely |
| high | stable | drift-warning | Investigate affected features and data quality |
| moderate or high | degraded | confirmed-degradation | Consider retraining |
| any | severe | confirmed-degradation | Immediate model review and controlled retraining |
| low | degraded | confirmed-degradation | Investigate possible concept drift or label/pipeline issues (performance dropped without strong input drift) |

Each recommendation text must include the evidence: top three features by DIS with their D, FI, S values, per-detector values, and (if available) the AUC drop with its confidence interval. Create an `alerts` row only when an alert type is not `none`, and avoid duplicate alerts for the same job and type.

## 14. MONITORING PIPELINE (`pipeline.py`)

`run_monitoring_job(model_id, batch_id, user_id)`:
1. Insert `monitoring_jobs` row (`status='running'`, window from the batch timestamps).
2. Load the model, reference profile, reference sample, thresholds, and the batch records from `prediction_logs`.
3. For each feature: PSI, KS (numerical), JSD, ADWIN; record durations.
4. Calibrate/normalise (Section 10), compute `D_i` and per-feature severity.
5. Compute SHAP importance and sensitivity on the batch; compute `DIS_i` and the ablation variants.
6. Compute conventional-alert flags.
7. If labels exist, compute performance state (Section 12).
8. Run the decision engine; create an alert if needed.
9. Write PostgreSQL: `drift_summaries` (one row per feature), `alerts`; update job `status='completed'`.
10. Write MongoDB: `drift_events` (per feature per detector, with histograms), `root_cause_reports`, `monitoring_logs` with timings.
11. On any exception: set job `status='failed'`, write the error to `monitoring_logs`, return a clear API error.

Run synchronously for batches up to `MAX_BATCH_ROWS`; expose `GET /monitoring/jobs/{job_id}` for status. All timing numbers come from real measurements.

## 15. RETRAINING AND DEPLOYMENT (`retraining.py`)

`POST /retrain` (administrator approval is required to deploy, any analyst can request):
1. Build the training set: original training split plus recent **labeled** production records (from `prediction_logs` where `true_label` is not null).
2. Data-quality validation (Section 8 report); stop with an error if it fails (for example too few labels or one class missing).
3. Split the recent labeled data: 70 percent added to training, 30 percent kept as the **held-out post-drift test set** (never used for training).
4. Train a challenger with the same algorithm (new version number), save as a model with status `challenger`.
5. Compare champion and challenger on the held-out post-drift set (AUC, F1, precision, recall).
6. Insert `retraining_events` with `decision='pending'` and the comparison in `models.metrics` of the challenger.
7. `POST /retrain/{event_id}/approve` (administrator only): deploy only if `challenger_auc >= champion_auc + RETRAIN_MIN_GAIN_AUC` and F1 is not worse; then, in **one transaction**, set the challenger to `champion`, the old champion to `retired`, and `decision='deployed'`. If criteria fail, return a 409 with the reason and set `decision='rejected'`.
8. `POST /retrain/{event_id}/reject` and `POST /retrain/{event_id}/rollback` (administrator only): rollback restores the previous champion, marks the deployed one `retired`, sets `rollback_flag=true`, `decision='rolled_back'`.

Never retrain automatically from a drift alert alone. The "approved model version" is what the production application would use via `GET /models/{dataset_id}/champion`.

## 16. SIMULATION MODULE (`app/simulation/`)

`DriftSpec` fields: `feature` (or `label_function` for concept drift), `kind` (`mean_shift`, `scale_change`, `proportion_change`, `concept_flip`), `magnitude`, `pattern` (`sudden`, `gradual`, `recurring`, `none`), `onset_batch`, `duration_batches` (for gradual), `period_batches` (for recurring).

Scenarios to implement as named presets in `scenarios.py`:
1. `income_shift`: income mean shifted by a chosen number of standard deviations.
2. `loan_amount_shift`: loan amounts scaled up.
3. `credit_score_shift`: credit score mean shifted down.
4. `employment_mix_change`: proportions of `employment_type` changed (more self-employed/unemployed).
5. Patterns: each of the above with `sudden`, `gradual` (linear interpolation from 0 to full magnitude over `duration_batches`), `recurring` (drift on and off in a period).
6. Controls: `no_drift`; `low_importance_drift` (drift in `num_credit_lines` or `home_ownership`); `concept_drift_only` (change the label coefficients, no change in the feature distributions).

Every generated batch comes with **ground truth metadata**: which features are drifting, magnitude, pattern, onset batch, and whether concept drift is active. Store this in `experiment_runs` for evaluation. The Simulation Lab page and `POST /simulation/run` must let the user pick a scenario, create N batches, push them through the normal ingest path (`/batches`) and optionally release labels with a delay.

## 17. EXPERIMENT RUNNER (`app/experiments/`)

`runner.py` runs, for a configurable list of scenarios and seeds (default 10 seeds), the full loop: train baseline, build reference profile, stream simulated batches (default 30 batches of 1,000 records, drift onset at batch 10), run the monitoring pipeline, and compare **DriftGuard** with the **conventional baseline** (Section 11.5). Write results to `experiments/results/` as CSV and PNG plots and also store summaries in `experiment_runs`.

Metrics (`metrics.py`), each computed from real runs only:
1. **Drift detection accuracy** (where ground truth exists): per-feature precision, recall, F1 of "feature flagged" versus "feature truly drifting", for both methods.
2. **False-positive rate**: batch-level alert rate on `no_drift` and `low_importance_drift` scenarios (a low-importance drift that does not hurt performance should ideally not trigger a high-priority DriftGuard alert).
3. **Detection delay**: batches between drift onset and the first alert.
4. **Feature-prioritisation quality**: for single-feature drift scenarios of equal standardised magnitude, measure the actual AUC drop for each feature (ground truth consequence), then compute Spearman rank correlation and top-k agreement between that ground-truth ranking and the ranking by (a) conventional PSI, (b) D only, (c) DIS and the other variants.
5. **Drift–performance relationship**: correlation between batch drift scores and AUC drop, and the confusion between warnings and confirmed degradation.
6. **Retraining benefit**: compare the policies "retrain on every conventional alert" and "retrain only on DriftGuard's consider-retraining/immediate-review recommendation": number of retrainings and AUC on a held-out post-drift test set.
7. **Computational overhead**: measured time per batch for each stage, memory if easily measured.

Report mean and a bootstrap 95 percent interval across seeds. The README must say that results appear only after running the experiments and must not contain pre-filled numbers.

## 18. REST API SPECIFICATION

All endpoints are under `/api/v1`, JSON in and out, OpenAPI docs at `/docs`. Auth: `Authorization: Bearer <JWT>`. Roles: `analyst`, `administrator`, `reviewer` (read-only). Return consistent error bodies `{ "detail": ..., "code": ... }` with correct HTTP codes (400, 401, 403, 404, 409, 422).

### 18.1 Auth
- `POST /auth/register` (administrator creates users; the very first user may self-register as administrator when the table is empty).
- `POST /auth/login` → `{access_token, token_type, role, name}`.
- `GET /auth/me`.

### 18.2 Datasets
- `POST /datasets` (multipart: `file` CSV, `name`, `source`, `target_column`) → stores the dataset file, infers feature types, returns data-quality report; inserts `datasets` and `features` rows.
- `GET /datasets`, `GET /datasets/{dataset_id}` (with features).

### 18.3 Models
- `POST /models` `{dataset_id, algorithm, params?}` → trains, builds reference profile, stores `models` row (status `champion` if first for the dataset).
- `GET /models?dataset_id=`, `GET /models/{model_id}` (metrics, status), `GET /models/{dataset_id}/champion`.
- `GET /models/{model_id}/reference` (reference profile summary without raw data).

### 18.4 Batches
- `POST /batches` `{model_id, records:[{record_id?, timestamp?, features:{...}}]}` or multipart CSV. Validates the schema against the reference features and types, predicts, writes `prediction_logs` (with `batch_id`) and a `production_batches` document. Returns `batch_id`, `n_rows`, and any generated `record_id`s.
- `GET /batches?model_id=`, `GET /batches/{batch_id}`.

### 18.5 Labels
- `POST /labels` `{model_id, labels:[{record_id, true_label}]}` → updates `prediction_logs`; rejects unknown records and non-0/1 labels; returns counts updated/skipped.

### 18.6 Monitoring
- `POST /monitoring/jobs` `{model_id, batch_id}` → runs the pipeline; returns `job_id`, status.
- `GET /monitoring/jobs?model_id=`, `GET /monitoring/jobs/{job_id}` (job + drift summaries + alert + performance state).
- `GET /monitoring/jobs/{job_id}/report` (the MongoDB root-cause report), `GET /monitoring/jobs/{job_id}/events` (detector-level documents).
- `POST /monitoring/jobs/{job_id}/reevaluate` (after labels arrive).
- `GET /monitoring/trend?model_id=&feature=` (per-batch series of D, DIS and AUC for charts).

### 18.7 Alerts
- `GET /alerts?model_id=&type=`, `GET /alerts/{alert_id}`.

### 18.8 Retraining
- `POST /retrain` `{model_id, alert_id?}`, `GET /retrain`, `GET /retrain/{event_id}` (with champion-vs-challenger comparison), `POST /retrain/{event_id}/approve|reject|rollback` (administrator only).

### 18.9 Thresholds (administrator)
- `GET /thresholds`, `PUT /thresholds/{name}` `{value}` (validates ranges and ordering, for example `severity_moderate < severity_high`; writes `threshold_history`), `GET /thresholds/history`.

### 18.10 Simulation and experiments
- `GET /simulation/scenarios`, `POST /simulation/run` `{model_id, scenario, n_batches, batch_size, seed, label_delay_batches}`.
- `POST /experiments/run` `{scenarios, seeds, n_batches}` (administrator; may run as a background task), `GET /experiments/{run_id}`, `GET /experiments/{run_id}/results`.

### 18.11 Export
- `GET /monitoring/jobs/{job_id}/export?format=csv|json`.

## 19. SECURITY AND VALIDATION

- **Passwords:** Argon2 hashes only; never log or return them. Minimum password length 8 with at least one letter and one digit.
- **JWT:** signed with `SECRET_KEY`, expiry from config; generic login error message ("Invalid email or password").
- **RBAC:** a dependency per endpoint. Analysts: datasets, models, batches, labels, monitoring, retrain request. Administrators: everything including users, thresholds, retrain approve/reject/rollback, experiments. Reviewers: GET only.
- **SQL injection:** only SQLAlchemy parameterised queries; no string-built SQL.
- **NoSQL injection:** build Mongo filters only from validated, typed Pydantic fields; reject any user-supplied key beginning with `$`.
- **Input validation (Pydantic + extra checks):** email format, name length, file extension `.csv` and size limit, required target column, schema must match reference features and types on `/batches`, unknown columns rejected, row limit `MAX_BATCH_ROWS`, labels must be 0/1 and refer to existing records, thresholds numeric and ordered, date windows ordered.
- **Rate limiting** on `/auth/login` and upload endpoints (for example `slowapi`).
- **Transport/secrets:** configuration through environment variables; `.env` is gitignored; document HTTPS termination as a deployment note.
- **Audit:** log logins, threshold changes, alert creation and retraining decisions with user id and time (PostgreSQL rows plus `monitoring_logs`).

## 20. FRONTEND (STREAMLIT) – NEAT, CLEAN UI

Multi-page Streamlit app using the API. Consistent layout, sidebar navigation, role-aware menu, readable fonts, meaningful labels, no raw tracebacks (show friendly messages). Use Plotly for charts. Pages:

1. **Home / Login**: login form; shows the user's role; sign-out.
2. **Datasets**: upload form with inline validation messages; table of datasets; data-quality report view.
3. **Models**: train a model (dataset, algorithm); list of models with status and metrics; reference profile summary.
4. **Monitoring**: choose model; upload or select a batch; button "Run monitoring"; job list with status.
5. **Drift Results**: for a selected job: ranked table (feature, PSI, KS stat and p, JSD, normalised D, FI, S, DIS, severity, detectors flagged); bar chart of DIS; reference-vs-production histogram overlays per feature; badge for **drift-warning** or **confirmed-degradation**; recommendation text with evidence; display of detector agreement/disagreement; a toggle to compare DIS variants.
6. **Performance**: AUC/F1 over batches with the baseline line and confidence band; table of labeled batches; button to submit/refresh labels and re-evaluate a job.
7. **Alerts and Retraining**: list of alerts with type and level; "Request retraining"; for administrators, a champion-vs-challenger table with Approve, Reject and Rollback buttons and clear confirmation dialogs.
8. **Simulation Lab**: choose scenario, pattern, magnitude, batches, label delay; run; view live progress and the resulting drift charts; ground-truth metadata shown next to detections.
9. **Experiments**: launch the experiment runner (admin), view results tables and plots, download CSV.
10. **Admin**: users (create/list), thresholds editor with history, system status (database connectivity).

Show an "illustrative / simulated data" banner whenever data comes from the simulator.

## 21. AUTOMATION

- `scheduler.py` (APScheduler): optional job that, every N minutes (config), finds batches without a monitoring job and runs them; and one that re-evaluates jobs when new labels are present. Disabled by default; enable by `ENABLE_SCHEDULER=true`.
- GitHub Actions workflow `.github/workflows/ci.yml`: install, lint (ruff), run unit tests with coverage.
- `Makefile` targets: `setup`, `up`, `down`, `seed`, `demo`, `test`, `lint`, `experiments`, `queries`.

## 22. TESTS (pytest)

Unit tests:
- PSI: identical samples give approximately 0; a known shifted sample gives a positive value and respects the `1e-4` clipping; categorical handling with unseen categories.
- KS: statistic and p-value agree with SciPy; categorical returns None.
- JSD: bounded in [0,1]; symmetric; 0 for identical distributions.
- Normalisation: value below the noise floor maps to 0; at or above the cap maps to 1; monotonic.
- SHAP aggregation: one-hot columns are summed back to their original feature; output shape handling for list and array return types.
- Sensitivity: an irrelevant feature has near-zero sensitivity compared with an important feature on a toy model.
- DIS: product and each variant on small hand-computed examples.
- Decision engine: every row of the Section 13 table.
- Performance state: stable, degraded and severe cases with synthetic labels.
- Simulation: drift patterns (sudden, gradual, recurring) produce the expected feature shifts; ground-truth metadata is correct.
- Security: password hashing round-trip; JWT expiry; role dependency denies correctly.

Integration tests (FastAPI TestClient with test databases):
- Full flow: register admin → login → upload dataset → train model → submit no-drift batch → run job (low drift, no alert) → submit drifted batch → run job (alert) → submit labels → re-evaluate (confirmed-degradation) → retrain request → approve → rollback.
- RBAC: reviewer cannot POST; analyst cannot approve retraining.
- Validation errors: wrong schema, bad labels, bad threshold ordering.
- Transaction: a failed deploy leaves exactly one champion.

## 23. QUERY DELIVERABLE (for the DBMS lab test)

Write `queries/sql_queries.sql` and `queries/mongo_queries.js`, runnable against the seeded demo data, each with a comment stating the question it answers. Also write `scripts/run_queries.py` to execute them and print results.

SQL (PostgreSQL), S1–S12: (S1) features of a dataset; (S2) top drifted features of a job by DIS with a join; (S3) average PSI per feature with GROUP BY and HAVING; (S4) features with DIS above the overall average using a subquery; (S5) top two features per job using a CTE and `RANK()`; (S6) alerts with job, model and user using a multi-table join; (S7) jobs without alerts using a LEFT JOIN; (S8) a view `v_high_drift`; (S9) change in DIS between consecutive jobs with `LAG()`; (S10) old versus new model per retraining event via a self-join on `models` and JSONB field extraction; (S11) deploy a challenger in a transaction; (S12) an index plus `EXPLAIN ANALYZE`.

MongoDB, N1–N9: (N1) unlabelled high-probability predictions with projection; (N2) count and mean probability per batch; (N3) accuracy per batch where labels exist; (N4) drift events above conventional thresholds, sorted; (N5) top feature per root-cause report with `$unwind`; (N6) `$lookup` between `drift_events` and `root_cause_reports`; (N7) update delayed labels; (N8) indexes and `explain("executionStats")`; (N9) retention delete of old logs.

## 24. DOCKER, SEED DATA AND README

- `docker-compose.yml` with services: `postgres:15` (mounts `db/init.sql`), `mongo:6` (mounts `db/mongo_init.js`), `api` (FastAPI via Uvicorn), `frontend` (Streamlit). Health checks and `depends_on`. Named volumes for the databases and `artifacts/`.
- `scripts/seed_demo.py`: creates three users (administrator, analyst, reviewer) with demo passwords printed to the console and documented in the README as **local demo credentials only**; generates the synthetic dataset; trains a model; creates reference profile; pushes several simulated batches (no drift, then drift); runs jobs; releases some labels; creates alerts and one retraining event. This data must make all queries in Section 23 return meaningful rows.
- `scripts/run_demo.py`: scripted 8-step demonstration against the running API: login, upload dataset, train model, submit no-drift batch, submit drifted batch, view drift ranking, submit labels and re-evaluate (warning becomes confirmed degradation), request and approve retraining with rollback available; then run two SQL and two MongoDB queries.
- `README.md`: project overview, architecture diagram (Mermaid), setup (`docker compose up`, local run without Docker), how to seed, how to run the demo, how to run tests, how to run experiments, API overview, folder map, the hybrid-database rationale, security notes, limitations, and an explicit statement that no experimental results are included until the experiments are executed. Include a section "Mapping to the DBMS lab components" (ER diagram, normalisation, DFD, queries).
- `docs/ARCHITECTURE.md`: five layers (ingestion, analysis, decision, storage, presentation) and the flow of one monitoring run showing exactly what is written to PostgreSQL and what to MongoDB.

## 25. BUILD PHASES (WITH ACCEPTANCE CHECKS)

1. **Foundation:** repo, config, logging, Docker compose, `init.sql`, `mongo_init.js`, SQLAlchemy models, DB connections, health endpoint. *Check:* `docker compose up` starts both databases and the API; `/health` reports both connected.
2. **Auth and RBAC:** users, login, JWT, role dependencies, rate limiting. *Check:* tests for the auth flow and denied roles pass.
3. **Data and ML core:** synthetic generator, dataset upload and data-quality report, preprocessing, training, reference profile, model endpoints. *Check:* a trained model with stored metrics and reference profile; tests pass.
4. **Drift engine:** PSI, KS, JSD, ADWIN, null calibration, normalisation, conventional baseline. *Check:* unit tests pass; known-shift synthetic data is flagged and no-drift data is mostly not.
5. **Explainability and DIS:** SHAP importance with aggregation, sensitivity, DIS and variants. *Check:* tests pass; an important feature gets a higher FI/S than an irrelevant one.
6. **Pipeline, performance and decisions:** batch ingest, labels, monitoring job, performance state, decision engine, alerts, reports, re-evaluation. *Check:* the integration flow up to confirmed-degradation passes.
7. **Retraining:** champion–challenger, approval, rejection, rollback with one transaction. *Check:* tests for exactly one champion and rollback pass.
8. **Simulation and experiments:** scenarios, patterns, ground truth, experiment runner, metrics, plots. *Check:* a short experiment run (2 seeds, 15 batches) completes and writes real CSV/PNG outputs.
9. **Frontend:** all Streamlit pages with validation messages and clean layout. *Check:* every page loads for each role; the demo script can be followed in the UI.
10. **Queries, seed, demo, docs:** `queries/`, `seed_demo.py`, `run_demo.py`, README, architecture docs, CI. *Check:* `make seed`, `make demo`, `make queries`, `make test` all succeed on a clean clone.

## 26. DEFINITION OF DONE (VERIFY AND REPORT ACTUAL OUTPUT)

- `docker compose up` brings up PostgreSQL, MongoDB, API and the Streamlit UI on a clean machine.
- `make seed` populates both databases; `make queries` runs S1–S12 and N1–N9 with non-empty, sensible output.
- `make test` passes with coverage reported; no skipped tests without a stated reason.
- The scripted demo shows: login by role, dataset upload with validation errors on a bad file, model training, no-drift batch giving no alert, drifted batch giving a ranked **drift-warning**, labels turning it into **confirmed-degradation**, retraining request with champion–challenger comparison, approval, and rollback.
- A short experiment run produces real result files; the README contains no invented numbers.
- All security items in Section 19 are implemented and covered by tests.
- The eight core table names and columns match Section 6.1 exactly; ER relationships match the foreign keys.
- `docs/DECISIONS.md` and `docs/KNOWN_LIMITATIONS.md` are complete and honest (for example: simulated drift only, tabular binary classification only, SHAP is model attribution and not causality, univariate detectors can miss multivariate drift, DIS thresholds need calibration).

When you finish, give me: the repository tree, the commands to run everything, the demo credentials, what was verified and how, and a list of anything not implemented or any assumption you made.