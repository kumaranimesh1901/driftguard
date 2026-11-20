# 07 · FastAPI Endpoints, Security & Dashboard Walkthrough

## 1. Complete API Route Reference (33 Routes from Live `/openapi.json`)

| Group | Method | Path | Role | Purpose |
|---|---|---|---|---|
| Health | `GET` | `/health` | Public | Returns `{status, postgres, mongo}` |
| Auth | `GET` | `/api/v1/auth/status` | Public | Checks whether any user exists (first-admin bootstrap) |
| Auth | `POST` | `/api/v1/auth/register` | Public (1st admin) / Admin | Creates user with Argon2 password hash (201 / 401 / 403 / 409) |
| Auth | `POST` | `/api/v1/auth/login` | Public (5/min) | Verifies password, returns HS256 JWT bearer token (200 / 401 / 429) |
| Auth | `GET` | `/api/v1/auth/me` | Any auth | Returns current user identity |
| Datasets | `POST` | `/api/v1/banking/datasets/validate` | Analyst+ | Validates CSV schema, types, missing values, preview |
| Datasets | `GET` | `/api/v1/banking/datasets/baseline` | Reviewer+ | Returns active baseline summary & champion model metrics |
| Datasets | `POST` | `/api/v1/banking/datasets/baseline` | Admin | Registers baseline CSV, profiles features, trains v1 XGBoost |
| Batches | `GET` | `/api/v1/banking/batches` | Reviewer+ | Lists all production batches with latest analysis summary |
| Batches | `POST` | `/api/v1/banking/batches/upload` | Analyst+ | Uploads one or more production batch CSVs |
| Batches | `GET` | `/api/v1/banking/batches/{batch_id}` | Reviewer+ | Returns batch detail and record preview |
| Analysis | `POST` | `/api/v1/banking/analysis/run` | Analyst+ | Runs individual or combined drift analysis (reuses unchanged signature) |
| Analysis | `POST` | `/api/v1/banking/analysis/run-all` | Analyst+ | Runs analysis on all unanalysed batches |
| Analysis | `GET` | `/api/v1/banking/analysis/history` | Reviewer+ | Returns `latest_per_batch`, `runs`, and `first_drift` milestones |
| Analysis | `GET` | `/api/v1/banking/analysis/latest/{batch_id}` | Reviewer+ | Latest full root-cause report for a batch |
| Analysis | `GET` | `/api/v1/banking/analysis/{job_id}` | Reviewer+ | Full root-cause report for a specific job |
| Analysis | `GET` | `/api/v1/banking/analysis/{job_id}/features/{feature}` | Reviewer+ | Per-feature detail & histogram |
| Analysis | `GET` | `/api/v1/banking/compare` | Reviewer+ | Direct A→B drift comparison between two batches |
| Views | `GET` | `/api/v1/banking/overview` | Reviewer+ | KPIs, DIS trend, heatmap, distributions, default rates |
| Views | `GET` | `/api/v1/banking/model-health` | Reviewer+ | Champion holdout + per-batch labelled production evaluation |
| Views | `GET` | `/api/v1/banking/recommendations/{job_id}` | Reviewer+ | Decision engine risk category, actions, evidence, `low_impact_drift_features` |
| Views | `GET` | `/api/v1/banking/reports/{job_id}/pdf` | Reviewer+ | Generates & streams ReportLab committee PDF |
| Views | `GET` | `/api/v1/banking/audit-logs` | Reviewer+ | Monitoring jobs (PostgreSQL) + audit events (MongoDB) |
| Simulation | `GET` | `/api/v1/banking/simulation/scenarios` | Reviewer+ | Describes Scenarios A, B, C, D |
| Simulation | `POST` | `/api/v1/banking/simulation/generate-baseline` | Admin | Generates synthetic 2020 baseline |
| Simulation | `POST` | `/api/v1/banking/simulation/generate-batch` | Analyst+ | Generates one synthetic scenario batch |
| Simulation | `POST` | `/api/v1/banking/simulation/seed-demo` | Admin | Seeds & analyses full 2020 + 2026 Jan/Mar/Jun/Sep/Nov demo |
| Retraining | `GET` | `/api/v1/banking/models` | Reviewer+ | Lists all model versions (`champion`/`challenger`/`retired`) |
| Retraining | `GET` | `/api/v1/banking/models/retraining` | Reviewer+ | Lists retraining events joined with Mongo `model_evaluations` |
| Retraining | `POST` | `/api/v1/banking/models/retrain` | Admin | Trains & evaluates challenger (labelled data only, 201 / 409) |
| Retraining | `POST` | `/api/v1/banking/models/retraining/{event_id}/deploy` | Admin | Promotes acceptable challenger in one PG transaction |
| Retraining | `POST` | `/api/v1/banking/models/retraining/{event_id}/reject` | Admin | Retires pending challenger without deploying |
| Retraining | `POST` | `/api/v1/banking/models/retraining/{event_id}/rollback` | Admin | Restores previous champion and sets `rollback_flag = true` |

---

## 2. Security Controls

- **Password hashing:** Argon2 (`argon2-cffi`) via `app/core/security.py`. Password policy enforced by Pydantic in `app/schemas/auth.py` (min 8 chars, letter + digit).
- **JWT authentication:** HS256 tokens expiring in `60` minutes (`settings.access_token_expire_minutes`). Outside `APP_ENV=dev`, `Settings.validate_production_secret()` refuses default/short `SECRET_KEY`.
- **RBAC:** Enforced on every endpoint via `Depends(require_roles(...))`.
- **Rate limiting:** `slowapi` limits `POST /auth/login` to `5/minute` per IP (`app/core/rate_limit.py`).
- **Upload validation:** Enforces `max_upload_mb = 50`, `max_batch_rows = 50000`, required banking columns, and regex `BATCH_ID_PATTERN`.
- **CORS (`app/main.py`):** Explicitly restricted to `["http://localhost:8501", "http://127.0.0.1:8501"]` (no wildcard `*` with credentials).
- **SQL injection prevention:** All PostgreSQL queries use SQLAlchemy 2.0 ORM (`select(...)`, bound parameters)—no raw string-interpolated SQL anywhere.

---

## 3. Dashboard Walkthrough (10 Pages — Zero UI Statistics)

All dashboard pages live in `frontend/sections/` and call `frontend/api_client.py` only. Shared styling and Plotly templates live in `frontend/theme.py` and `frontend/components.py`. Screenshots are saved in `~/DriftGuard_extras/docs/screenshots/`.

| Page | API Endpoints Used | What to Show & Say in the Demo |
|---|---|---|
| **Login** (`dashboard.py`) | `/health`, `/auth/status`, `/auth/login` | Show strict login gating (`st.navigation` hides all app pages without a JWT) and sign in as Administrator. |
| **Overview** (`overview.py`) | `/overview`, `/batches`, `/analysis/latest/{id}`, `/models`, `/audit-logs` | Point to the single-batch hero (`November_2026`, `CRITICAL`, `DIS 1.000`, `Severe degradation`), the 4-row Drift Types panel, 5 KPI cards, DIS trend with labelled threshold bands, Top 5 `evidence_row` mini-bars, and Recent Alerts. |
| **Batch Monitoring** (`batch_monitoring.py`) | `/batches`, `/datasets/validate`, `/batches/upload`, `/compare`, `/audit-logs` | Show the sortable batch ledger with DIS progress bars, drag-and-drop CSV upload, population A-vs-B comparison, and combined pooling. |
| **Customer Drift Analysis** (`drift_analysis.py`) | `/batches`, `/analysis/latest/{id}` | Show `ProgressColumn` bars for D/FI/S/DIS, `detector_agreement`, ADWIN detection caption (record 78), and the feature distribution overlay vs bootstrap noise floors. |
| **Drift Impact Score** (`drift_impact.py`) | `/analysis/history`, `/analysis/{job_id}` | Show the `D × FI × S = DIS` diagram, `"Why is Debt-to-Income ranked #1?"` box, sorted DIS chart, ablation table, and the honest experiment note on `D×FI` vs `S`. |
| **Model Health** (`model_health.py`) | `/model-health`, `/models` | Show Champion card, Production ROC-AUC (95% CI) error-bar chart vs baseline dashed line (`September` stable vs `November` severe), confusion matrices, and the unlabelled empty state on `March_2026`. |
| **Model Versions & Retraining** (`model_versions.py`) | `/models`, `/models/retraining`, `/models/retrain`, `/{id}/deploy\|reject\|rollback` | Show the `v1 → v5` version timeline, champion-vs-challenger comparison with CI bar & ✓/✗ criteria, and `@st.dialog` confirmation modals. |
| **Risk Recommendations** (`recommendations.py`) | `/batches`, `/recommendations/{job_id}` | Show per-batch risk cards, action checklists, affected-feature chips, and the separate `"Check data quality"` callout for low-impact strong drift. |
| **Reports** (`reports.py`) | `/analysis/history`, `/reports/{job_id}/pdf`, `/audit-logs` | Compile and download a committee PDF report and show the audit log entry. |
| **Dataset Management** (`datasets.py`) | `/datasets/baseline`, `/batches`, `/simulation/*` | Show baseline schema table, upload validation, and the smart synthetic batch form (auto-prefilled ID + scenario mismatch & future-date warnings). |
