# DRIFTGUARD: INDEPENDENT VERIFICATION AUDIT REPORT

**Date of Audit:** 2026-11-19  
**Auditor:** Independent Automated Verification Agent  
**Audit Target:** DriftGuard Repository (`/Users/animesh/Documents/5th SEM/DBMS_PROJECT/code`)  
**Specification References:** `docs/SPEC.md`, `docs/KNOWN_LIMITATIONS.md`, CD252IA Lab Rubric  

---

## 1. VERDICT

**Status:** **INCOMPLETE** (Phase 1–2 foundation and core statistical math present; Phases 3–10 largely unbuilt).  
**Single Biggest Risk:** The repository lacks the core monitoring pipeline, model retraining, database write workflows, REST routers (datasets, models, batches, jobs), Streamlit frontend, and query deliverables (S1–S12, N1–N9), meaning the application cannot be demonstrated end-to-end or evaluated against the course grading rubric in its current state.  
**Honesty Assessment:** High integrity in existing documentation (`docs/KNOWN_LIMITATIONS.md` transparently records unimplemented modules; no fabricated metric numbers or inflated claims were discovered).

---

## 2. SUMMARY TABLE BY SECTION

| Section | Audit Domain | Status | Key Evidence / Reason |
|---|---|---|---|
| **1. Repository Structure** | Directory Layout & Files | `PARTIAL` | Missing `queries/`, `scripts/`, `frontend/`, `app/experiments/`, subpackage `__init__.py` files, and 10 of 11 API routers. |
| **2. Static Quality** | Linters, Stubs, & Syntax | `PASS` | `ruff` and `black` pass cleanly on existing code; `compileall` succeeds; no TODO/FIXME/stubs found. |
| **3. Unit & Integration Tests** | Test Coverage & Regressions | `PARTIAL` | 13 unit tests pass (58% coverage). Only 5 of 16 named regression tests exist; integration tests absent. |
| **4. Database Conformance (PG)** | Schema, Constraints, & ORM | `PARTIAL` | `db/init.sql` matches 8 core + 2 supporting tables; ORM models in `app/db/models.py` lack `Threshold`, `ThresholdHistory`, CHECK constraints, and partial unique index. Live PG `NOT VERIFIED`. |
| **5. MongoDB Conformance** | Collections, Indexes, & Writes | `PARTIAL` | `db/mongo_init.js` and `app/db/mongo.py` define collections/indexes; however, zero write paths exist in `app/`. Live Mongo `NOT VERIFIED`. |
| **6. Drift Engine Correctness** | Numerical Audit of Detectors | `PARTIAL` | PSI, KS, JSD, scalar Normalisation, and DIS formula verified numerically. `adwin_monitor.py`, `shap_importance.py`, `sensitivity.py`, `performance.py`, and detector aggregation missing. |
| **7. API Audit** | REST Endpoints, Auth, & RBAC | `PARTIAL` | Only 3 of 31 spec endpoints exist (`/auth/register`, `/auth/login`, `/auth/me`, plus `/health`). Auth and RBAC work as designed for implemented routes; rate limiting and 422 error wrapping missing. |
| **8. End-to-End Flow** | 8-Step Lifecycle Execution | `FAIL` | Ingestion, training, batch monitoring, label submission, and retraining routers do not exist; flow cannot execute past Step 1. |
| **9. Query Deliverable** | S1–S12 & N1–N9 Execution | `FAIL` | Files `queries/sql_queries.sql`, `queries/mongo_queries.js`, and `scripts/run_queries.py` do not exist. |
| **10. Experiments** | Multi-Seed Runner & Metrics | `FAIL` | Runner, evaluation metrics, and plots modules are missing; `make experiments` target absent. |
| **11. Frontend** | Multi-Page Streamlit UI | `FAIL` | Entire `frontend/` directory is missing (`Home.py`, `pages/`, `api_client.py`). |
| **12. Security Checklist** | Hashes, RBAC, Tokens, Injection | `PARTIAL` | Argon2id, JWT expiry/iat, secret enforcement, and parameterized SQL verified; rate limiting, Mongo `$` validation, and audit logging missing. Git log `NOT VERIFIED` (not a git repo). |
| **13. Documentation & Honesty** | Spec Alignment & No Fakes | `PASS` | No fabricated metrics, no forbidden claims ("state of the art", "first system"); `docs/KNOWN_LIMITATIONS.md` is strictly accurate. |
| **14. User Commands** | External Verification Scripts | `PASS` | Exact docker, psql, mongosh, and make reproduction commands supplied in Section 5 below. |
| **15. Lab Rubric Coverage** | CD252IA Academic Mapping | `PARTIAL` | Relational schema and normalisation ready; query deliverables, forms/UI, and full experiment pipeline missing. |

---

## 3. FINDINGS LIST (SORTED BY SEVERITY)

### High Severity

- **[FIND-01] High: Core API Routers and Services Missing**
  - **File & Line:** `app/api/`, `app/services/`
  - **Evidence:** Only `auth.py` router exists. Missing endpoints: `/datasets`, `/models`, `/batches`, `/labels`, `/monitoring/jobs`, `/alerts`, `/retrain`, `/thresholds`, `/simulation/run`, `/experiments/run`, `/export`.
  - **Suggested Fix:** Implement the 10 missing routers in `app/api/` and orchestrator services (`preprocessing.py`, `training.py`, `pipeline.py`, `retraining.py`).

- **[FIND-02] High: Complete Frontend UI Missing**
  - **File & Line:** `frontend/Home.py`, `frontend/pages/`
  - **Evidence:** `ls frontend/Home.py` returns `No such file or directory`. The Streamlit 10-page application specified in Section 20 does not exist.
  - **Suggested Fix:** Implement Streamlit UI components using `requests` calling the API with role-based navigation.

- **[FIND-03] High: DBMS Lab Deliverables S1–S12 and N1–N9 Missing**
  - **File & Line:** `queries/sql_queries.sql`, `queries/mongo_queries.js`, `scripts/run_queries.py`
  - **Evidence:** `ls queries scripts` returns `No such file or directory`. `make queries` fails immediately.
  - **Suggested Fix:** Create `queries/sql_queries.sql` with S1–S12 and `queries/mongo_queries.js` with N1–N9, plus execution runner `scripts/run_queries.py`.

- **[FIND-04] High: Experiment Runner and Analysis Pipeline Missing**
  - **File & Line:** `app/experiments/runner.py`, `metrics.py`, `plots.py`
  - **Evidence:** `make experiments` target is absent from `Makefile`; directory `app/experiments/` is empty/missing.
  - **Suggested Fix:** Build multi-seed experiment runner streaming simulated batches and recording evaluation metrics to `experiments/results/` and MongoDB `experiment_runs`.

- **[FIND-05] High: Rate Limiting Not Implemented on Authentication Endpoints**
  - **File & Line:** `app/api/auth.py:68`
  - **Evidence:** 50 rapid failed login requests all returned HTTP 401 with zero HTTP 429 rate limit responses. `slowapi` is in `requirements.txt` but unused.
  - **Suggested Fix:** Initialize `Limiter(key_func=get_remote_address)` in `app/main.py` and decorate `@router.post("/login")` with `@limiter.limit("5/minute")`.

### Medium Severity

- **[FIND-06] Medium: Decision Engine Uses Stale Disclaimer on Stable Performance (R3 Regression)**
  - **File & Line:** `app/services/decision.py:27-30`
  - **Evidence:** When `level in {"moderate", "high"}` and `performance == "stable"`, `decide()` returns: `"Investigate and monitor closely; labels needed to confirm impact"`. Because labels are already available to verify performance stability, the disclaimer `"; labels needed to confirm impact"` contradicts Section 13.
  - **Suggested Fix:** Check `performance == "stable"` explicitly to return `"Investigate and monitor closely"` or `"Investigate affected features and data quality"`.

- **[FIND-07] Medium: ORM Models Missing `Threshold` and `ThresholdHistory`**
  - **File & Line:** `app/db/models.py:112`
  - **Evidence:** `db/init.sql` lines 9-10 define tables `thresholds` and `threshold_history`. `app/db/models.py` has no corresponding SQLAlchemy model classes.
  - **Suggested Fix:** Add `Threshold` and `ThresholdHistory` Mapped classes to `app/db/models.py`.

- **[FIND-08] Medium: ORM Models Lack CHECK Constraints and Partial Unique Index**
  - **File & Line:** `app/db/models.py:23-112`
  - **Evidence:** The 9 SQL CHECK constraints from `init.sql` (e.g. `role IN (...)`, `decision IN (...)`, `old_model_id <> new_model_id`) and the partial index `one_champion_per_dataset` are not defined in SQLAlchemy metadata.
  - **Suggested Fix:** Add `CheckConstraint` and `Index("one_champion_per_dataset", "dataset_id", unique=True, postgresql_where=(status == 'champion'))` to `__table_args__`.

- **[FIND-09] Medium: Missing Package `__init__.py` Files in `app/` Subdirectories**
  - **File & Line:** `app/api/`, `app/db/`, `app/schemas/`, `app/services/`, `app/services/drift/`, `app/services/explain/`, `app/simulation/`
  - **Evidence:** None of these 7 package directories contain `__init__.py`, violating Section 4 layout requirements.
  - **Suggested Fix:** Create empty `__init__.py` in all package directories under `app/`.

- **[FIND-10] Medium: Pydantic 422 Validation Errors Do Not Match Prescribed JSON Schema**
  - **File & Line:** `app/main.py:30`
  - **Evidence:** Sending invalid JSON produces FastAPI's default list-style `{"detail": [{"type": "...", "loc": [...]}]}` instead of `{"detail": "...", "code": "..."}`.
  - **Suggested Fix:** Add `@app.exception_handler(RequestValidationError)` returning `JSONResponse(status_code=422, content={"detail": str(exc), "code": "validation_error"})`.

- **[FIND-11] Medium: Missing MongoDB `$` Key Injection Validator**
  - **File & Line:** `app/schemas/`
  - **Evidence:** No schema or validator checks for or rejects MongoDB operator injection keys starting with `$`.
  - **Suggested Fix:** Implement a recursive validator rejecting dictionary keys matching `^\$`.

- **[FIND-12] Medium: Missing User Login and Operational Audit Logging**
  - **File & Line:** `app/api/auth.py:68`
  - **Evidence:** Login events do not record audit entries to PostgreSQL or `monitoring_logs` in MongoDB.
  - **Suggested Fix:** Log successful and failed authentication attempts with client timestamp and IP.

- **[FIND-13] Medium: Normalised Drift Aggregation Function Missing**
  - **File & Line:** `app/services/drift/normalise.py:1-8`
  - **Evidence:** `normalise.py` only implements single-value normalisation. Aggregation across detectors (`max` or `mean` per Section 10.2) is not implemented.
  - **Suggested Fix:** Add `aggregate_d(dn_dict: dict[str, float], method: str = "max") -> float`.

### Low Severity

- **[FIND-14] Low: README Mentions Queries Directory That Does Not Exist**
  - **File & Line:** `README.md:23`
  - **Evidence:** `README.md` states: *"SQL and Mongo query deliverables are in `queries/`"*, but directory `queries/` is absent.
  - **Suggested Fix:** Update README or restore `queries/` with deliverables.

- **[FIND-15] Low: Missing Files Referenced in Prompt and Specs**
  - **File & Line:** `.env.local.example`, `AGENTS.md`, `docs/FIX_ROUND2.md`
  - **Evidence:** These files are referenced in prompt instructions but absent from disk.
  - **Suggested Fix:** Add `.env.local.example` and `AGENTS.md` to repository root.

- **[FIND-16] Low: Repository Root Is Not Initialized as a Git Repository**
  - **File & Line:** Workspace root (`.`)
  - **Evidence:** `git status` returns `fatal: not a git repository (or any of the parent directories): .git`.
  - **Suggested Fix:** Run `git init`, commit tracking files, and ensure `.env` remains gitignored.

---

## 4. REAL NUMBERS OBSERVED

### 4.1 Unit Test Execution & Coverage
```bash
pytest -q --cov=app --cov-report=term-missing
```
**Output:**
```
.............                                                            [100%]
================================ tests coverage ================================
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
app/__init__.py                       0      0   100%
app/config.py                        40      1    98%   51
app/logging_config.py                 3      3     0%   3-8
app/main.py                          42     42     0%   3-60
app/security.py                      46     46     0%   3-99
app/services/decision.py              8      2    75%   7, 32
app/services/drift/jsd.py             6      0   100%
app/services/drift/normalise.py       4      1    75%   7
app/services/drift/psi.py            34      8    76%   13, 26, 50-57
app/services/explain/dis.py          15      2    87%   12, 14
app/simulation/generator.py          77     10    87%   32, 34, 36, 123, 127, 142, 144, 148, 150, 175
---------------------------------------------------------------
TOTAL                               275    115    58%
13 passed, 1 warning in 0.89s
```
- Total test count: **13 passed** (0 failures, 0 skips).
- Modules under 70% coverage: `app/logging_config.py` (0%), `app/main.py` (0%), `app/security.py` (0%).
- Unexercised modules in standard suite: `app/api/auth.py`, `app/db/models.py`, `app/db/mongo.py`, `app/db/postgres.py`, `app/services/drift/ks.py`, `app/simulation/scenarios.py`.

### 4.2 Numerical Drift Engine Checks (`tests/audit/run_drift_audit.py`)
- **PSI identical samples:** `0.000000e+00` (exact zero)
- **PSI +1 std shift:** `0.9732`
- **PSI +3 std shift:** `6.1866` (monotonic shift verified: `6.1866 > 0.9732`)
- **Binned proportions sum:** `p.sum() = 1.000000`, `q.sum() = 1.000000`
- **Binary categorical PSI (50% -> 90% shift):** `0.8789`
- **3-level categorical PSI:** `0.6227`
- **Zero-inflated feature PSI:** `1.7581`
- **Constant feature shift PSI:** `18.4188`
- **Adversarial NaN input:** Automatically dropped; returns finite PSI (`0.0000`).
- **Adversarial sample size (< 30 rows):** Raises `ValueError("at least 30 non-null observations are required")`.
- **KS statistic:** `0.231000` (difference vs SciPy `ks_2samp`: `0.00e+00`).
- **KS p-value:** `8.470621e-24` (difference vs SciPy `ks_2samp`: `0.00e+00`).
- **Categorical KS return:** `(None, None)` as required.
- **JSD(a, b):** `0.153561`; **JSD(b, a):** `0.153561` (exact symmetry: `True`).
- **JSD identical samples:** `0.000000e+00`.
- **JSD SciPy equivalence:** Exactly equals `jensenshannon(p, q, base=2)**2`.
- **Normalisation below noise floor (0.02 vs floor 0.05):** `0.0`.
- **Normalisation at noise floor (0.05):** `0.0`.
- **Normalisation intermediate (0.275, cap 0.50):** `0.5000`.
- **Normalisation at cap (0.50):** `1.0`.
- **Normalisation above cap (0.80):** `1.0`.
- **Bootstrap null noise floor (q95, 200 trials):** `0.0317`.
- **Synthetic baseline default rate:** `0.1704` (within required `[0.13, 0.22]`).
- **Toy DIS hand calculation match:**
  - `income`: `dis = 0.1200`, `d_only = 0.4000`, `d_fi = 0.2400`, `weighted_sum = 0.4750`, `rank_based = 0.6667`.
  - All factors bounded in `[0, 1]`: `True`.

### 4.3 REST API Audit (`tests/audit/run_api_audit.py`)
- Endpoints tested against Section 18: **38 routes tested**.
- Implemented routes: **3** (`POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`), plus `GET /health`.
- Missing routes: **35** (all returned HTTP 404).
- Bootstrap first-user admin requirement: Non-admin rejected with `403 Forbidden` (`{"code": "forbidden"}`). Admin created with `201 Created`.
- Anonymous registration after bootstrap: `401 Unauthorized`.
- Duplicate email registration: `409 Conflict`.
- Invalid password / unknown email login: Generic `401 Unauthorized` (`{"code": "invalid_credentials"}`).
- Role authorization: Analyst attempting admin action returns `403 Forbidden` (`{"code": "forbidden"}`).
- Bearer token authentication:
  - Missing token: `401 Unauthorized`.
  - Malformed/garbage token: `401 Unauthorized`.
  - Expired token: `401 Unauthorized`.
  - Deleted user token: `401 Unauthorized`.
- `/health` without database services running: Returns HTTP 503 `{"status": "degraded", "postgres": "unavailable", "mongo": "unavailable"}` without leaking server hostnames or connection strings.

---

## 5. NOT VERIFIED & USER COMMANDS LIST

Because Docker, PostgreSQL 15, and MongoDB 6 are not running inside this execution sandbox, the following items are marked **NOT VERIFIED** and require local execution:

### Commands to Run on Host Machine:

```bash
# 1. Environment and Container Startup
cp .env.example .env
docker compose up --build -d

# 2. Service Connectivity Verification
curl -i http://localhost:8000/health
# Expect: HTTP/1.1 200 OK {"status": "ok", "postgres": "connected", "mongo": "connected"}

# 3. Fault-Tolerance Check (Stop MongoDB)
docker compose stop mongo
curl -i http://localhost:8000/health
# Expect: HTTP/1.1 503 Service Unavailable {"status": "degraded", "postgres": "connected", "mongo": "unavailable"}
docker compose start mongo

# 4. PostgreSQL DDL and Constraint Verification
docker compose exec postgres psql -U driftguard -d driftguard -c "\d+ users"
docker compose exec postgres psql -U driftguard -d driftguard -c "\d+ models"
docker compose exec postgres psql -U driftguard -d driftguard -c "\d+ drift_summaries"
docker compose exec postgres psql -U driftguard -d driftguard -c "\d+ retraining_events"
docker compose exec postgres psql -U driftguard -d driftguard -c "\d+ thresholds"

# 5. Champion Partial Unique Index Verification
docker compose exec postgres psql -U driftguard -d driftguard -c "SELECT indexname, indexdef FROM pg_indexes WHERE indexname='one_champion_per_dataset';"

# 6. MongoDB Collection & Index Verification
docker compose exec mongo mongosh driftguard --eval "db.getCollectionNames()"
docker compose exec mongo mongosh driftguard --eval "db.prediction_logs.getIndexes()"
docker compose exec mongo mongosh driftguard --eval "db.production_batches.getIndexes()"
docker compose exec mongo mongosh driftguard --eval "db.root_cause_reports.getIndexes()"

# 7. Seed, Queries, and Demo Pipeline (Once implemented)
make seed
make queries
make demo
make test
```

---

## 6. HONESTY AND COMPLIANCE CHECK

- **Pre-filled or fabricated metrics:** None detected. Search for regex patterns resembling hardcoded accuracy/AUC across `app/`, `docs/`, and `README.md` found only configuration defaults (`NULL_QUANTILE=0.95`, `RETRAIN_MIN_GAIN_AUC=0.01`).
- **Forbidden Claims:** Checked `grep -rniE "first system|state of the art|novel|outperforms|proven|accuracy of"`. Zero marketing exaggerations found; `docs/SPEC.md:39` expressly forbids these claims.
- **DIS Status:** DIS is consistently referred to as a "proposed research component" across `app/services/explain/dis.py` and `docs/DECISIONS.md`.
- **Drift vs Degradation:** Explicitly separated in `app/services/decision.py` and documented in `README.md` and `docs/DECISIONS.md`.

---

## 7. AUDIT REPRODUCTION COMMANDS

The exact commands executed during this independent audit:
```bash
# Static quality checks
ruff check app tests
black --check app tests
python3 -m compileall -q app tests
grep -rn "TODO\|FIXME\|NotImplemented\|pass  #" app
grep -rnE ";\s*\w+\s*=" app
grep -rniE "password\s*=\s*['\"]|secret" app

# Test suite execution
pytest -q --cov=app --cov-report=term-missing

# Numerical drift audit
PYTHONPATH=. python3.11 tests/audit/run_drift_audit.py

# Adversarial API and RBAC audit
PYTHONPATH=. python3.11 tests/audit/run_api_audit.py

# Clean up temporary test files
# Note: Audit scripts preserved in tests/audit/ for inspection:
#   - tests/audit/run_drift_audit.py
#   - tests/audit/run_api_audit.py
```
