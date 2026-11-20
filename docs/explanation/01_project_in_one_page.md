# 01 · DriftGuard in One Page

## 1. Spoken Pitches (Word-for-Word Scripts)

### One-Sentence Pitch
> **DriftGuard** is a dual-database (PostgreSQL + MongoDB) credit-risk monitoring platform that detects input data drift, prioritises which drifted features actually matter to the deployed ML model, separates drift warnings from label-confirmed performance drops, and manages human-approved champion–challenger retraining with rollback.

### 30-Second Version
> "When a bank deploys a loan-default prediction model, customer data changes over time—incomes rise, debt ratios grow, or credit scores shift. Traditional drift tools fire alerts on every changing column, even columns the model barely uses, which causes alert fatigue. DriftGuard solves this by combining three statistical detectors—PSI, KS, and JSD—with the model's own SHAP feature importance to rank which shifts actually matter. Crucially, DriftGuard never assumes drift means accuracy dropped: it waits for real loan outcomes before confirming degradation, and supports administrator-approved champion–challenger retraining backed by PostgreSQL transactions and MongoDB telemetry."

### 2-Minute Version
> "My project is **DriftGuard**, a database-backed monitoring framework for a credit-risk ML model built with FastAPI, Streamlit, PostgreSQL 15, and MongoDB 6.
>
> Here is the core problem: a bank trains an XGBoost loan-default classifier on 2020 customers and deploys it in 2026. Over time, applicant distributions shift—that is **data drift**. Standard monitoring tools check each feature independently with PSI or KS tests and alert on everything. That creates two failures: first, **alert fatigue**, because shifting a low-importance feature like `age` does not affect loan decisions the same way shifting `debt_to_income` does; second, **premature retraining**, because input drift does *not* automatically mean model accuracy dropped.
>
> DriftGuard addresses both. For every incoming production batch, it runs PSI, KS, and JSD against bootstrap sampling noise floors (`app/services/drift/calibration.py`), normalises them into a drift score $D \in [0, 1]$, and combines $D$ with TreeSHAP feature importance ($FI$) and permutation sensitivity ($S$) into a Drift Impact Score, $\text{DIS} = D \times FI \times S$. It also runs River's ADWIN streaming detector on the model's predicted probabilities and error stream (`app/services/drift/adwin.py`).
>
> For governance, DriftGuard strictly separates **drift warnings** from **confirmed degradation** (`app/services/recommendations/decision.py`). Without outcome labels, degradation is reported as *Unknown – labels pending*. Only when at least 200 labelled loan outcomes arrive and the bootstrap 95% confidence interval of ROC-AUC drops below baseline does it raise `confirmed-degradation`. An administrator can then train a challenger model on labelled data only, evaluate it on a 40% independent post-drift test split, and atomically deploy or roll it back in PostgreSQL.
>
> From a DBMS perspective, structured metadata, constraints, alerts, and model lifecycles live in 10 normalised PostgreSQL tables—including a partial unique index enforcing one champion per dataset—while large flexible records, histograms, ADWIN stream events, and full reports live in 8 MongoDB collections."

### 5-Minute Version (Structured Walkthrough)
1. **Motivation (45s):** Explain the 2020 → 2026 loan-default scenario. Emphasise that **data drift ≠ performance degradation** (a bank's applicants can get wealthier without breaking the model's ranking ability; conversely, **concept drift** can break accuracy even if input distributions barely move).
2. **Architecture & Hybrid DBMS (75s):** Walk through Streamlit (`frontend/dashboard.py`) → FastAPI (`app/api/banking.py`) → PostgreSQL (`db/init.sql`) + MongoDB (`db/mongo_init.js`). Explain *why* two databases: ACID transactions, foreign keys, CHECK constraints, and `one_champion_per_dataset` in PostgreSQL vs unbounded JSON records, histograms, and audit logs in MongoDB.
3. **Drift & Model-Aware Prioritisation (75s):** Explain PSI, KS, JSD, bootstrap noise-floor calibration (fixed during audit from n-vs-n to N-vs-n), TreeSHAP ($FI$), permutation sensitivity ($S$), $\text{DIS} = D \times FI \times S$, and ADWIN on prediction/error streams.
4. **Decision Engine & Retraining (60s):** Show how `decide()` in `app/services/recommendations/decision.py` maps `(drift_level, performance_state)` to `LOW / MEDIUM / HIGH / CRITICAL`. Walk through our live demo: January (`LOW`) → March (`MEDIUM`) → June (`HIGH`) → September (`HIGH`, labelled *stable*!) → November (`CRITICAL`, labelled *severe*). Show champion–challenger retraining on `November_2026` (+0.270 AUC gain on 640 independent test rows) with one-transaction deploy and rollback.
5. **Honest Research Findings & Limitations (45s):** Present the 10-seed, 70-batch counterfactual experiment (`scripts/experiments/dis_prioritisation.py`): model-aware weighting ($D \times FI$, NDCG@3 = `0.932`) and DIS (`0.910`) clearly beat conventional PSI ranking (`0.774`), while permutation sensitivity $S$ added no measurable gain over $D \times FI$. Also note that no drift score correlated with real AUC drop (Spearman `-0.180` to `-0.063`), proving labels are essential. Acknowledge synthetic data and single model family (XGBoost).

---

## 2. The Problem: Data Drift vs Model Degradation

Imagine a bank trains a credit-risk model on 2020 applicants to predict `loan_default` (0 = repaid, 1 = defaulted) from 12 attributes (`age`, `monthly_income`, `credit_score`, `credit_limit`, `outstanding_loan_amount`, `fd_balance`, `debt_to_income`, `loan_amount_requested`, `loan_tenure`, `number_of_existing_loans`, `interest_rate`, `employment_type`).

- **Data drift (covariate shift):** $P(X)$ changes—for example, in 2026 inflation raises `monthly_income` by 22% (Scenario B) or borrowers carry higher `debt_to_income` (Scenario C/D).
- **Concept drift:** $P(y \mid X)$ changes—the *relationship* between inputs and default changes (e.g., an economic shock makes previously safe borrowers default, as in `November_2026`).
- **Why "drift ≠ the model got worse":** In our live demo, `September_2026` has severe input drift (`Overall DIS = 1.000`, `7/12` features drifted), yet its 1,800 labelled outcomes confirm **stable** model performance (`ROC-AUC` does not drop). Retraining blindly on `September_2026` would waste effort. Conversely, `November_2026` has concept shift and its AUC collapses (`performance_state = severe`), genuinely justifying controlled retraining.

---

## 3. The 7 Questions DriftGuard Answers

| # | Question | How DriftGuard Answers It | Code Reference |
|---|---|---|---|
| 1 | **Has the input data changed?** | Compares production batch against 2020 baseline using PSI, KS, JSD + bootstrap noise floors | `app/services/drift/psi.py`, `ks.py`, `jsd.py`, `calibration.py` |
| 2 | **Which features changed?** | Evaluates all 12 monitored features individually and reports `detector_agreement` | `app/services/monitoring/analysis.py → compute_feature_drift()` |
| 3 | **How much did they change?** | Subtracts the 95th-percentile noise floor and scales to $D_i \in [0, 1]$ (`max` aggregation) | `app/services/drift/normalise.py → normalise()`, `aggregate_d()` |
| 4 | **Do those features matter to the model?** | Weights $D_i$ by TreeSHAP importance ($FI_i$) and permutation sensitivity ($S_i$) into $\text{DIS}_i$ | `app/services/monitoring/model.py → shap_importance()`, `permutation_sensitivity()`; `app/services/explain/dis.py → dis_variants()` |
| 5 | **Is model performance actually dropping?** | Requires ≥ 200 labels; computes bootstrap 95% CI of ROC-AUC vs baseline + ADWIN error stream | `app/services/monitoring/model.py → performance_state()`; `app/services/drift/adwin.py → adwin_stream()` |
| 6 | **What should the risk team investigate?** | Maps `(drift_level, performance_state)` to `LOW / MEDIUM / HIGH / CRITICAL`, action checklist, and `low_impact_drift_features` | `app/services/recommendations/decision.py → decide()`; `risk.py → risk_assessment()` |
| 7 | **Should we retrain, and is the new model better?** | Admin-triggered challenger trained on labelled rows only, tested on 40% holdout with paired bootstrap CI, atomic deploy & rollback | `app/services/monitoring/retraining.py → create_challenger()`, `deploy()`, `reject()`, `rollback()` |

---

## 4. Central Research Question & Honest Answer

**Exact wording** (from `FINAL_PROJECT_REPORT.md`, Section 1.3):
> *"Can combining multi-detector statistical drift magnitude with model-specific feature importance and observed empirical performance changes improve the prioritisation of potentially consequential drift compared with conventional unweighted drift alerts?"*

**Honest answer from our experiment** (`scripts/experiments/dis_prioritisation.py`, 10 seeds, 70 covariate-shift ranking batches, 110 total batches):
1. **Yes for model-aware weighting ($D \times FI$):** Weighting normalised drift by SHAP importance raises `NDCG@3` from **`0.774`** (conventional PSI) to **`0.932`** ($D \times FI$) and **`0.910`** (full DIS = $D \times FI \times S$).
2. **No for permutation sensitivity ($S$):** $D \times FI$ matched or beat full DIS in **88.6%** of batches because $S$ correlates strongly with $FI$ (Spearman `0.867–0.930` on our demo batches). Keeping $S$ in the UI is transparent, and replacing it with *shift-aware* sensitivity is documented as future work.
3. **Labels remain mandatory for degradation:** Batch-level drift scores had near-zero/negative correlation with actual AUC drop (Spearman `-0.180` to `-0.063`), confirming that drift alerts cannot replace labelled performance checks.
