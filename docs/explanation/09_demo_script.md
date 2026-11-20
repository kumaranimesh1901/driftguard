# 09 · Live Demo Script (10–12 Minutes)

## 1. Pre-Demo Checklist (5 Minutes Before Viva)

1. Open **Docker Desktop** and wait until the engine is running.
2. Start the stack:
   ```bash
   cd /Users/animesh/projects/DriftGuard
   ./start.sh
   ```
3. Verify health in terminal:
   ```bash
   curl -s http://localhost:8000/health
   # Expected: {"status":"ok","postgres":"connected","mongo":"connected"}
   ```
4. Open three browser tabs:
   - Tab 1: `http://localhost:8501` (DriftGuard Dashboard — sign in as **Administrator**; confirm **"Hide test batches (`UI_Test_*`)"** is ON in the sidebar)
   - Tab 2: `http://localhost:8000/docs` (FastAPI Swagger UI)
   - Tab 3: Terminal ready for `psql` and `mongosh` queries.

---

## 2. Step-by-Step Spoken Walkthrough (~11 Minutes)

### Step 1 · Login & Role-Based Security (1 min)
- **Click:** Show the Login screen (`http://localhost:8501`), click **Administrator (Full)**.
- **Say:** *"Without a valid HS256 JWT token, `st.navigation` hides every application page. Passwords are hashed with Argon2 and `/auth/login` is rate-limited to 5 attempts per minute. The sidebar footer shows live health dots from `/health` for PostgreSQL, MongoDB, and FastAPI."*

### Step 2 · Overview — Single-Batch Hero & Progression (2 min)
- **Click:** Stay on **Overview**. Point to the Hero Banner, Drift Types panel, 5 KPI cards, DIS Trend chart, Top 5 Features mini-bars, and Recent Alerts.
- **Say:** *"Here is our 2020 baseline vs five 2026 production batches totalling 7,100 customers. The DIS Trend shows the exact lifecycle: `January_2026` is LOW RISK (DIS 0.000), `March_2026` is MEDIUM RISK (0.145, income +22%), `June_2026` and `September_2026` are HIGH RISK (1.000), and `November_2026` escalates to CRITICAL RISK because true default labels arrived and confirmed severe degradation. Notice `September_2026` is HIGH drift but its performance is STABLE—proving drift does not automatically mean degradation."*

### Step 3 · Customer Drift Analysis & ADWIN (1.5 min)
- **Click:** Sidebar → **Customer Drift Analysis**. Keep `November_2026` selected in the top-right global batch selector; scroll to the `ProgressColumn` feature table and select `Debt-to-Income` in the Feature Detail Inspector. Then switch the header selector to `March_2026`.
- **Say:** *"Every feature shows PSI, KS, JSD against its 95th-percentile bootstrap noise floor, normalised D, SHAP importance FI, sensitivity S, DIS, and whether the detectors agree. Above the table, ADWIN from River detected prediction-stream drift at production record 78 (`0.353 → 0.559`). When I switch to unlabelled `March_2026`, notice Concept Drift and Confirmed Degradation read **Unknown – labels pending**, never 'No'."*

### Step 4 · Drift Impact Score & Honest Research Finding (1.5 min)
- **Click:** Sidebar → **Drift Impact Score** (select `November_2026`).
- **Say:** *"This page decomposes $\text{DIS} = D \times FI \times S$ and explains why `Debt-to-Income` ranks #1. On the right is our ablation comparison. In our 10-seed counterfactual experiment, weighting drift by SHAP importance ($D \times FI$, NDCG@3 = 0.932) clearly beat conventional PSI ranking (0.774), whereas permutation sensitivity $S$ added no gain because it correlates 0.87–0.93 with $FI$."*

### Step 5 · Model Health — Labels Confirm Degradation (1.5 min)
- **Click:** Sidebar → **Model Health**.
- **Say:** *"The Production ROC-AUC chart plots bootstrap 95% confidence intervals against the dashed baseline line. `September_2026` stays on the baseline line (stable despite high drift), while `November_2026` drops well below it (confirmed severe degradation). Below are the side-by-side confusion matrices."*

### Step 6 · Champion–Challenger Retraining & Rollback (1.5 min)
- **Click:** Sidebar → **Model Versions & Retraining**. Expand Event #1 (`rolled_back`) or Event #4 (`deployed`).
- **Say:** *"DriftGuard never retrains automatically. An administrator trains a challenger using **only labelled rows** (0 unlabelled rows used), holding out 40% as an independent post-drift test set. Event #1 on `November_2026` gained +0.270 AUC (95% CI `[0.222, 0.336]`) with no baseline regression. Deploying runs in a single PostgreSQL transaction using `session.flush()` to satisfy the `one_champion_per_dataset` partial unique index, and Rollback restores the prior champion."*

### Step 7 · Live DBMS Queries (PostgreSQL + MongoDB) & PDF Report (2 min)
- **Click:** Switch to Terminal and run 2 PostgreSQL queries (Q2 top-5 JOIN + Q5 retraining self-JOIN with JSONB) and 1 MongoDB aggregation (M4). Then click **Reports** in the dashboard and click **Compile PDF Report** → **Download PDF**.
- **Say:** *"PostgreSQL enforces BCNF-style relational integrity, CHECK constraints, and atomic transactions; MongoDB stores nested histograms, ADWIN stream events, and model evaluations. Finally, the Reports page compiles a committee-ready PDF directly from the backend."*

---

## 3. Backup Plan If Anything Fails Live

- **Dashboard or API stops responding:** Run `./stop.sh && ./start.sh` (takes ~10 seconds; all Docker volumes are preserved).
- **Docker or network issue:** Open the pre-captured `1440px` PNG screenshots in `~/DriftGuard_extras/docs/screenshots/` (`overview.png`, `drift_analysis.png`, `drift_impact.png`, `model_health.png`, `model_versions.png`, `reports.png`) and walk through the exact same script.
