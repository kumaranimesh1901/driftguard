# 10 · Viva Questions & Model Answers (90 Questions)

Each question is tagged **[Easy]**, **[Medium]**, or **[Hard]** with a concise, honest answer grounded in DriftGuard's real code and measured numbers.

---

## Section A · Project Basics & Problem Motivation (15 Questions)

1. **[Easy] What problem does DriftGuard solve?**  
   It monitors a deployed credit-risk ML model (`XGBClassifier`) when incoming customer data changes over time. Instead of firing unranked alerts on every changing feature, it prioritises features by model impact (`DIS`), waits for true labels before confirming accuracy loss, and manages administrator-approved retraining.
2. **[Easy] What is the difference between data drift and concept drift?**  
   Data drift (covariate shift) means the input distribution $P(X)$ changed (e.g., applicant incomes rose 22% in `March_2026`). Concept drift means the relationship $P(y \mid X)$ changed—borrowers with the same attributes now default at a different rate (as in `November_2026`).
3. **[Easy] Why does "data drift" not automatically mean the model got worse?**  
   A model can still rank risky vs safe borrowers accurately even when inputs shift—for example, `September_2026` has `Overall DIS = 1.000` (7/12 features drifted) yet its 1,800 labelled outcomes confirm **stable** ROC-AUC. Only real labels can confirm performance degradation.
4. **[Easy] What is alert fatigue in ML monitoring?**  
   Conventional rules (`PSI > 0.25`, `KS p < 0.01`, `JSD > 0.1`) flag every shifted column equally, including low-importance columns like `age` or `fd_balance`. Weighting by model importance (`D × FI` / `DIS`) raised alert precision from `0.935` to `1.000` in our experiment.
5. **[Easy] What domain and dataset does DriftGuard use?**  
   Synthetic banking credit-risk data (`app/services/synthetic_data.py`): a 2,500-row 2020 baseline and 2026 production batches with 12 monitored features (11 numerical, 1 categorical `employment_type`) and binary target `loan_default`.
6. **[Easy] Why is the UI built in Streamlit while all logic is in FastAPI?**  
   Keeping all statistics, ML, and database writes inside FastAPI (`app/`) ensures one authoritative backend, enforces RBAC/JWT security on every call, and lets the Streamlit frontend (`frontend/api_client.py`) stay a thin presentation layer that computes zero statistics.
7. **[Easy] What are the 5 demo batches and what do they show?**  
   `January_2026` (Scenario A, `LOW RISK`, DIS 0.000), `March_2026` (Scenario B, `MEDIUM RISK`, DIS 0.145), `June_2026` (Scenario C, `HIGH RISK`, DIS 1.000), `September_2026` (Scenario D with labels, `HIGH RISK`, performance `stable`), and `November_2026` (Scenario D + concept shift with labels, `CRITICAL RISK`, performance `severe`).
8. **[Medium] What happens when a batch has no `loan_default` labels?**  
   DriftGuard still computes PSI, KS, JSD, $D$, $FI$, $S$, DIS, and ADWIN prediction drift, but sets `performance_state = "unknown"` and `confirmed_degradation = None` / `concept_drift_suspected = None` (displayed as *Unknown – labels pending*, never "No").
9. **[Medium] Why did you separate "prediction drift" from "data drift" and "concept drift"?**  
   Because they have different causes and remedies (`analysis.py → classify_drift_types()`). Data drift means inputs shifted; prediction drift (ADWIN/KS on predicted probabilities) means model outputs shifted; suspected concept drift means error rose while input drift stayed low; confirmed degradation requires labelled AUC drop.
10. **[Medium] Does DriftGuard ever retrain or deploy models automatically?**  
    No. Both `decision.py` and `retraining.py` enforce human-in-the-loop governance: only an administrator can trigger `POST /models/retrain`, inspect the independent test evaluation, and explicitly call `/deploy`, `/reject`, or `/rollback`.
11. **[Medium] What is the `UI_Test_ScenarioC` batch and why was it hidden?**  
    MongoDB logs showed it was generated during a UI test with Scenario A (*Stable population*, 1,200 rows) and default future date `2026-12-31`, which legitimately gave DIS `0.000` and shadowed `November_2026`. Rather than deleting DB records, we added a sidebar toggle **"Hide test batches (`UI_Test_*`)"** (default ON).
12. **[Medium] How does DriftGuard avoid recomputing identical analyses?**  
    `app/api/banking.py → _run()` hashes `(mode, baseline_sha256, model:{champion_id}, settings_fingerprint(), batch_sha256s)`. If a completed report matches and `force=False`, `store.find_reusable_report()` returns it (`reused=True`).
13. **[Medium] What happens if a feature drifts strongly (`high` D) but has tiny DIS?**  
    `risk.py → risk_assessment()` keeps model risk low, but adds the feature to `low_impact_drift_features`, recommends checking upstream data quality/pipelines, and never tells the user "distributions remain stable".
14. **[Hard] Why not just retrain the model on a fixed monthly schedule?**  
    Scheduled retraining wastes compute and risks deploying a worse model when no concept shift occurred (e.g., `September_2026` was stable despite high drift), and it cannot train safely if true default labels haven't matured yet. DriftGuard retrains only when justified and only if the challenger beats the champion on an independent test set.
15. **[Hard] Can DriftGuard monitor a regression model or image data?**  
    **Not implemented.** The current codebase (`app/services/datasets.py`, `model.py`) is built specifically for tabular binary classification (`XGBClassifier`) with the 12 banking attributes. Extending to regression or unstructured data is future work.

---

## Section B · Drift Statistics & ADWIN (15 Questions)

16. **[Easy] What does PSI measure and what is its formula?**  
    Population Stability Index (`app/services/drift/psi.py`) measures binned distribution shift: $\text{PSI} = \sum (p_k - q_k)\ln(p_k / q_k)$, using 10 baseline quantile bins with open-ended outer edges `(-inf, +inf)` and `1e-4` clipping.
17. **[Easy] Why are the outer PSI bins set to `(-inf, +inf)`?**  
    So production values smaller than the 2020 minimum or larger than the 2020 maximum are counted in the first/last bin instead of being silently dropped by `np.histogram()`.
18. **[Easy] How does DriftGuard handle categorical features like `employment_type`?**  
    `categorical_psi()` adds an explicit unseen-category bucket (reference proportion `0.0` clipped to `1e-4`), `jsd_from_counts()` computes JSD on category shares, and `ks()` returns `(None, None)` because categories have no numeric order.
19. **[Easy] What does the KS test measure?**  
    `app/services/drift/ks.py → ks()` wraps `scipy.stats.ks_2samp` to compute the maximum vertical distance between the reference and production empirical CDFs ($D_{\text{KS}} \in [0, 1]$) plus its p-value.
20. **[Easy] Why use JSD in addition to PSI and KS?**  
    JSD (`app/services/drift/jsd.py → jsd_from_counts()`) is symmetric ($\text{JSD}(P\|Q)=\text{JSD}(Q\|P)$) and strictly bounded in $[0, 1]$ (using base-2 log, squaring SciPy's JS distance), whereas PSI is unbounded.
21. **[Medium] Why do you require at least 30 finite rows in `_clean()`?**  
    With fewer than 30 values, 10-bin histograms and KS statistics become too noisy to interpret; `compute_feature_drift()` catches the `ValueError` and marks the feature `"Insufficient data"` with $D = 0$.
22. **[Medium] What is the difference between KS statistic and KS p-value?**  
    `ks_stat` is the **effect size** (magnitude of shift); `ks_p` is statistical significance. With large $n$ (e.g., 10,000 rows), even a negligible shift yields $p < 0.01$. That is why DriftGuard normalises `ks_stat` above its noise floor rather than trusting $p < 0.01$ alone.
23. **[Medium] What is a bootstrap noise floor (`calibration.py`)?**  
    It is the 95th percentile (`settings.null_quantile = 0.95`) of PSI, KS, and JSD across `b = 200` no-drift pairs drawn with replacement from the baseline. Any production statistic at or below the floor normalises to `0.0`.
24. **[Medium] Explain the noise-floor bug found during the audit and how you fixed it.**  
    Originally `_draw_pair()` drew two batch-sized ($n$ vs $n$) samples, inflating variance ($\sim 2/n$ vs $1/n + 1/N$) and giving ~0.55% (PSI) / 0.85% (KS) false alarms instead of 5%. Fixing it to draw ($N$ vs $n$) with replacement calibrated false alarms to ~7.35% / 4.10% around the nominal 5%.
25. **[Medium] Why does `aggregate_d()` use `max` instead of `mean`?**  
    PSI, KS, and JSD capture different shift shapes (bin tails vs median shift). Taking `max` ensures a real shift caught by one detector isn't diluted, while `detector_agreement` (`agree-drift`, `disagree`, `agree-stable`) makes any disagreement transparent.
26. **[Medium] What are the normalisation caps and severity thresholds in `config.py`?**  
    Caps: `norm_cap_psi = 0.5`, `norm_cap_ks = 0.5`, `norm_cap_jsd = 0.3`. Feature severity: `low` ($D < 0.15$), `moderate` ($0.15 \le D < 0.50$), `high` ($D \ge 0.50$).
27. **[Medium] What is ADWIN and how does DriftGuard use it?**  
    ADWIN (Adaptive Windowing, `river.drift.ADWIN(delta=0.002)` in `app/services/drift/adwin.py`) detects mean shifts in a bounded numeric stream. DriftGuard runs it on two model streams: `prediction` ($p \in [0,1]$, no labels) and `error` ($|y-p|$, labelled only), prepended with the shuffled champion holdout as reference.
28. **[Medium] Why is ADWIN *not* used for feature-distribution drift?**  
    ADWIN tracks the running mean of a single bounded stream; it is not a histogram/CDF test for multi-modal feature shifts and assumes a stream order. PSI/KS/JSD handle feature distributions; ADWIN handles model output and error streams.
29. **[Hard] Why do you shuffle the ADWIN reference segment with a fixed seed?**  
    The baseline holdout has no natural time order, so shuffling with `settings.random_seed = 42` makes it stationary by construction; any detection inside the reference segment (`n_detections_reference`) acts as a sanity check (ideally `0`).
30. **[Hard] Can univariate detectors miss multivariate drift?**  
    Yes. If `monthly_income` and `loan_amount_requested` keep their exact marginal distributions while their correlation reverses, univariate PSI/KS/JSD can miss it. Multivariate tests (MMD / classifier two-sample test) are documented as future work (**Not implemented**).

---

## Section C · ML Model, SHAP, Sensitivity & DIS (15 Questions)

31. **[Easy] Which ML algorithm does DriftGuard train and why?**  
    `XGBClassifier` (`n_estimators=250, max_depth=4, learning_rate=0.05`) inside a scikit-learn `Pipeline` (`app/services/monitoring/model.py`). Gradient-boosted trees perform strongly on tabular banking data and support exact, fast `TreeExplainer` SHAP values.
32. **[Easy] How do you handle missing values and unseen categories in the ML pipeline?**  
    Numerical columns use `SimpleImputer(strategy="median")`; categorical columns use `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`.
33. **[Easy] How is class imbalance handled?**  
    `fit_pipeline()` sets `scale_pos_weight = (len(y_tr) - pos) / pos`, weighting defaults proportionally to non-defaults, and `select_threshold()` picks the F1-maximising threshold on `[0.05, 0.95]` using the **training split only**.
34. **[Easy] Why select the decision threshold on the training split instead of the holdout?**  
    Tuning the threshold on the holdout would leak holdout information into the decision rule and make holdout F1/accuracy optimistically biased.
35. **[Easy] What are the baseline v1 model's real holdout metrics?**  
    On the 25% stratified holdout (625 rows): `ROC-AUC = 0.664`, `Accuracy = 0.760`, `F1 = 0.342`, confusion matrix `[[436, 61], [89, 39]]`.
36. **[Hard] Why is your baseline ROC-AUC only ~0.66 instead of 0.90+?**  
    Because our synthetic banking generator (`app/services/synthetic_data.py`) adds realistic stochastic noise to loan defaults (credit default in real banking is inherently noisy, where AUCs of 0.65–0.75 are typical). We did not fabricate an inflated 0.85+ AUC; the monitoring framework works identically at any baseline AUC.
37. **[Easy] What is SHAP and how is $FI$ computed?**  
    SHAP attributes each prediction to input features using Shapley values (`shap.TreeExplainer`). `shap_importance()` computes $\text{mean}(|\text{SHAP}|)$ on up to 2,000 production rows, sums one-hot columns back to `employment_type`, and divides by the max so $FI_i \in [0, 1]$.
38. **[Medium] Why sum one-hot SHAP values back to the original feature?**  
    Because SHAP values are additive in log-odds space; summing `cat__employment_type_*` attributions gives the total contribution of `employment_type` so it aligns 1-to-1 with its drift score $D$.
39. **[Medium] Does SHAP prove causal impact?**  
    No. SHAP explains how the *trained XGBoost model* uses features (attribution), not real-world causality.
40. **[Medium] How is permutation sensitivity $S$ computed?**  
    `permutation_sensitivity()` shuffles column $i$ across the batch `5` times (`settings.sensitivity_repeats`), computes $\text{mean}(|\Delta \hat{p}|)$, and divides by the maximum so $S_i \in [0, 1]$.
41. **[Hard] Why does $S$ exist in the formula if your experiment showed it doesn't help?**  
    The initial design hypothesised that $S$ would capture non-linear sensitivity beyond SHAP $FI$. Our empirical experiment tested that hypothesis honestly and found $S$ correlates `0.87–0.93` with $FI$, so $D \times FI$ (`NDCG@3 = 0.932`) slightly outperformed $D \times FI \times S$ (`0.910`). We kept `DIS = D × FI × S` intact for comparability and documented redefining $S$ as *shift-aware* sensitivity as future work.
42. **[Easy] Write the DIS formula and explain why it is multiplicative.**  
    $\text{DIS}_i = D_i \times FI_i \times S_i \in [0, 1]$. Multiplication acts like a logical AND: if drift $D_i = 0$ (no shift) OR importance $FI_i = 0$ (model ignores the feature), $\text{DIS}_i = 0$. An additive weighted sum fails this property.
43. **[Medium] Why do June, September, and November all show `Overall DIS = 1.000`?**  
    Because `Overall DIS = max(DIS_i)` and `debt_to_income` has `FI = 1.0`, `S = 1.0`, and PSI > 0.50 cap (`1.12` to `1.94`) in all three batches, saturating $D = 1.0$ and $\text{DIS} = 1.0$. The UI pairs Overall DIS with drifting-feature counts (`5/12` vs `7/12`) and raw PSI/KS/JSD.
44. **[Medium] What happens to DIS if the baseline has no `loan_default` labels?**  
    No model can be trained, so `dis_mode` falls back to `"drift_only"` ($FI = 1, S = 1$, so $\text{DIS} = D$) and the dashboard displays a warning banner.
45. **[Medium] What are the 4 DIS ablation variants in `dis.py`?**  
    `d_only` ($D$), `d_fi` ($D \times FI$), `weighted_sum` ($0.5D + 0.25FI + 0.25S$), and `rank_based` (normalised mean rank across $D, FI, S$).

---

## Section D · Database Design & DBMS Concepts (20 Questions)

46. **[Easy] Why did you use *both* PostgreSQL and MongoDB instead of one database?**  
    PostgreSQL enforces strict relational integrity (ACID transactions, FKs, CHECKs, `one_champion_per_dataset`) for structured metadata, alerts, and model lifecycles. MongoDB handles high-volume, variable-schema documents (10,800 customer records, nested histograms, ADWIN arrays, full root-cause JSON reports) without rigid migrations.
47. **[Easy] How many tables are in PostgreSQL and how many collections are in MongoDB?**  
    10 tables in PostgreSQL (`db/init.sql`)—8 actively written by the app plus 2 reserved (`thresholds`, `threshold_history`)—and 8 collections in MongoDB (`db/mongo_init.js`).
48. **[Easy] What are the primary keys and foreign keys in `drift_summaries`?**  
    PK is `summary_id` (`SERIAL`); FKs are `job_id → monitoring_jobs(job_id) ON DELETE CASCADE` and `feature_id → features(feature_id)`, plus `UNIQUE(job_id, feature_id)`.
49. **[Easy] What M:N relationship exists in your schema and how is it resolved?**  
    `monitoring_jobs` ↔ `features` is Many-to-Many (one job measures 12 features; each feature is measured in many jobs), resolved by the associative entity `drift_summaries`.
50. **[Easy] What `CHECK` constraints exist in `db/init.sql`?**  
    `users.role IN ('analyst','administrator','reviewer')`, `datasets.row_count >= 0`, `features.type IN ('numerical','categorical')`, `models.status IN ('champion','challenger','retired')`, `monitoring_jobs.window_end >= window_start`, `drift_summaries.severity IN ('low','moderate','high')`, `alerts.alert_type IN ('drift-warning','confirmed-degradation')`, `retraining_events.decision IN (...)`, and `old_model_id <> new_model_id`.
51. **[Medium] Explain how your schema satisfies 1NF, 2NF, and 3NF.**  
    1NF: atomic rows with surrogate PKs, no repeating feature columns. 2NF: no partial dependencies on composite keys (`drift_summaries` resolves M:N). 3NF: no transitive dependencies (user/feature/model attributes live only in their parent tables).
52. **[Medium] Is there any deliberate denormalisation in your schema?**  
    Yes: `drift_summaries.severity` could be recomputed from `D` and thresholds, but storing it preserves the exact historical severity judgment even if `.env` thresholds change later.
53. **[Medium] What is the partial unique index `one_champion_per_dataset`?**  
    `CREATE UNIQUE INDEX one_champion_per_dataset ON models(dataset_id) WHERE status='champion'`. Unlike a normal `UNIQUE(dataset_id, status)`, it allows unlimited `retired` or `challenger` models per dataset while guaranteeing **at most one** `champion`.
54. **[Medium] Why do `deploy()` and `rollback()` call `session.flush()` between updating the two models?**  
    Without `session.flush()`, SQLAlchemy might emit the `new -> champion` UPDATE before `old -> retired`, violating `one_champion_per_dataset`. Flushing sends `old -> retired` first inside the same uncommitted transaction, then promotes `new`, then commits atomically.
55. **[Medium] Why does `features` have `ON DELETE CASCADE` from `datasets`?**  
    A feature's reference statistics belong exclusively to its parent baseline dataset; if a dataset is deleted, orphaned feature rows make no sense, so cascading maintains referential integrity.
56. **[Medium] Why do `models` and `features` use `JSONB` instead of plain `TEXT`?**  
    `JSONB` stores binary-parsed JSON, validates syntax on write, and allows SQL extraction operators like `(metrics->>'roc_auc')::numeric` (used in SQL Q5/Q10) and GIN indexing if needed.
57. **[Medium] Explain ACID properties in the context of DriftGuard's `deploy()` transaction.**  
    **Atomicity:** both model status flips and `retraining_events.decision` commit together or roll back together. **Consistency:** `CHECK` and `one_champion_per_dataset` are preserved. **Isolation:** concurrent readers never see zero or two champions mid-swap. **Durability:** once committed, the new champion survives container restarts via the `pg_data` Docker volume.
58. **[Medium] How are PostgreSQL and MongoDB linked?**  
    By application-level foreign references: MongoDB documents store PostgreSQL integer IDs (`job_id`, `dataset_id`, `model_id`, `event_id`, `user_id`), indexed with unique indexes (`root_cause_reports.job_id`, `model_evaluations.event_id`) for fast O(log N) lookups.
59. **[Hard] What happens if MongoDB fails halfway through writing a monitoring job?**  
    In `app/services/monitoring/store.py → complete_job()`, all Mongo writes (`root_cause_reports`, `drift_events`, `stream_events`) and `session.commit()` sit inside one `try/except`. On any failure, PostgreSQL rolls back, MongoDB deletes all documents for that `job_id`, and `fail_job()` marks the job `failed`—leaving zero orphan reports (verified by `test_failed_mongo_event_write_leaves_no_orphan_report`).
60. **[Medium] What is the difference between `INNER JOIN` and `LEFT JOIN` in your demo queries?**  
    In Q2 (`drift_summaries JOIN features`), every summary has a feature so `INNER JOIN` is exact. In Q7 (`monitoring_jobs LEFT JOIN alerts`), stable jobs like `January_2026` have *no* alert row, so `LEFT JOIN` keeps them and displays `'none'` via `COALESCE`.
61. **[Medium] How does your `drift_summaries_job_id` index speed up queries?**  
    It creates a B-tree index on `job_id` so filtering `WHERE job_id = 14` jumps directly to the 12 matching rows instead of scanning the entire table.
62. **[Medium] Give an example of a self-join in DriftGuard.**  
    SQL Q5 joins `retraining_events` to `models` **twice** (`mo` on `old_model_id` and `mn` on `new_model_id`) to compare the old champion's AUC against the challenger's AUC in one row.
63. **[Medium] Give an example of a MongoDB aggregation pipeline in DriftGuard.**  
    Query M4 runs `$group` on `drift_events` by `$feature` to compute `avg_dis` and count conventional alerts (`$sum` with `$cond`), then `$sort` descending and `$limit: 5`.
64. **[Easy] Why are `thresholds` and `threshold_history` currently empty (0 rows)?**  
    They were designed in `db/init.sql` for future dashboard threshold editing (**Not implemented**); currently all thresholds are read from `.env` via `app/core/config.py`, and `settings_fingerprint()` invalidates cached analyses whenever `.env` changes.
65. **[Hard] Why didn't you use Two-Phase Commit (2PC) between PostgreSQL and MongoDB?**  
    MongoDB and PostgreSQL do not share a native XA/2PC coordinator. Using a compensating transaction (saga pattern) in `complete_job()` is simpler, avoids distributed lock blocking, and guarantees clean rollback in both stores on failure.

---

## Section E · Backend, API & Security (8 Questions)

66. **[Easy] How are passwords stored?**  
    Hashed with Argon2 (`argon2-cffi`) in `app/core/security.py → hash_password()`. Plaintext passwords are never stored or logged.
67. **[Easy] How does JWT authentication work?**  
    `POST /api/v1/auth/login` verifies the Argon2 hash and returns an HS256-signed JWT (`sub`, `role`, `iat`, `exp = +60 min`). Clients send `Authorization: Bearer <token>`; `current_user()` validates signature, expiry, and loads the user.
68. **[Easy] How is RBAC enforced?**  
    Via FastAPI dependency injection `Depends(require_roles("administrator"))` (or `"analyst"`, `"reviewer"`). Unauthorized roles receive HTTP 403 (`forbidden`).
69. **[Easy] How do you prevent SQL injection?**  
    All PostgreSQL access goes through SQLAlchemy 2.0 ORM (`select(...)` with bound parameters)—never string-formatted SQL.
70. **[Medium] How do you prevent brute-force login attacks?**  
    `slowapi` (`app/core/rate_limit.py`) enforces `5/minute` per IP on `POST /api/v1/auth/login`, returning HTTP 429 on excess requests.
71. **[Medium] Why did you restrict CORS in `app/main.py`?**  
    Previously `allow_origins` had `"*"` alongside `allow_credentials=True`, which is unsafe. It now allows only `http://localhost:8501` and `http://127.0.0.1:8501`.
72. **[Medium] What happens if someone tries to register as the first user with role `analyst`?**  
    `app/api/auth.py → register()` rejects it with HTTP 403 (`"First user must be administrator"`). After that, only an authenticated admin can create users.
73. **[Medium] How does `settings_fingerprint()` prevent stale cached analyses?**  
    `store.py → settings_fingerprint()` hashes all 22 result-affecting settings in `config.py` into `cfg:<16 hex>` and includes it in `input_signature`, so changing any threshold in `.env` automatically forces fresh computation.

---

## Section F · Retraining & MLOps Governance (7 Questions)

74. **[Easy] Can unlabelled production rows ever be used for retraining?**  
    Never. `retraining.py → _labelled_production()` filters `df[LABEL_COLUMN].notna()` and refuses with HTTP 409 if fewer than `200` labelled rows exist; `model_evaluations` explicitly logs `unlabelled_rows_used: 0`.
75. **[Easy] How is the post-drift test set kept independent?**  
    `create_challenger()` splits labelled production records 60% train / 40% stratified test (`POST_DRIFT_TEST_SHARE = 0.4`). The 40% test set is never used for fitting or threshold selection.
76. **[Medium] What are the 3 criteria a challenger must pass before deployment is allowed?**  
    (1) `AUC gain >= 0.01` (`retrain_min_gain_auc`), (2) paired-bootstrap 95% CI lower bound `> 0`, and (3) challenger baseline-holdout AUC `>= champion baseline AUC - 0.05` (no catastrophic forgetting).
77. **[Medium] What happens if an admin tries to deploy a challenger that failed the criteria?**  
    `retraining.py → deploy()` refuses with HTTP 409 (`"unvalidated models are never deployed"`), and the dashboard disables the Approve & Deploy button.
78. **[Medium] What is `leakage_warning` in a monitoring report?**  
    After v2+ is deployed, `app/api/banking.py → _run()` checks whether any analysed `batch_id` overlaps `metrics["training_batches"]`. If so, it sets `leakage_warning` and the dashboard warns that performance on that batch is optimistic.
79. **[Medium] After deploying v2, does the drift reference population change?**  
    No. The drift reference stays the registered 2020 baseline; only the active model (SHAP FI, sensitivity S, performance reference AUC, ADWIN holdout reference) updates to v2. Re-baselining is an explicit separate action.
80. **[Hard] Can two challengers be pending at the same time for the same champion?**  
    No. `create_challenger()` checks PostgreSQL `retraining_events` and refuses with HTTP 409 if a `pending` challenger already exists for the current champion.

---

## Section G · Research Experiment & Honest Limitations (10 Questions)

81. **[Hard] Is DIS your invention?**  
    Combining statistical drift with SHAP feature importance is a known idea in recent ML monitoring literature, not a brand-new theorem. Our specific contribution is the calibrated noise-floor formulation, the end-to-end dual-database governance platform, and the empirical ablation showing what actually works ($D \times FI$) vs what doesn't ($S$).
82. **[Hard] Your data is synthetic, so how do you know it works on real banking data?**  
    We do not claim generalisation to all real-world banks—that is an explicit limitation in our report. Synthetic data lets us control exact ground-truth shifts (covariate vs concept shift) and test every failure mode reproducibly; validating on a real historical credit dataset is future work.
83. **[Medium] How did you define "ground truth" in the DIS prioritisation experiment?**  
    For each feature $i$, we undid only feature $i$'s drift via rank-preserving quantile mapping back onto the baseline marginal and measured `mean |f(X) - f(X_undone)|` and decision flip rate.
84. **[Medium] What were the headline NDCG@3 numbers from your experiment?**  
    Across 70 covariate-shift batches (10 seeds): `conventional_psi = 0.774`, `d_only = 0.829`, `dis = 0.910`, and `d_x_fi = 0.932` (best overall).
85. **[Medium] What did the experiment show about alert precision and recall?**  
    DIS alerts (`DIS >= 0.05`) achieved `1.000` precision (`196/196` true positives) vs `0.935` for conventional alerts (`272/291`), at lower recall (`0.396` vs `0.549`)—a classic threshold trade-off.
86. **[Hard] Why did batch-level DIS have a negative correlation (`-0.106`) with real AUC drop?**  
    Because our 110-batch pool includes both pure covariate-shift batches (high DIS `1.0`, zero AUC drop) and concept-shift batches (low/moderate DIS, large AUC drop). That proves mathematically why input drift cannot substitute for labelled performance monitoring.
87. **[Medium] How many automated tests does DriftGuard have and what do they cover?**  
    **77 pytest tests** (all passing, `0` failed, `0` skipped) covering exact math vs hand calculations (`test_drift_methods.py`), auth/RBAC/JWT/rate-limiting (`test_api_auth.py`), full banking workflow (`test_banking_platform.py`), and audit regressions including noise-floor calibration, ADWIN, partial-write compensation, CORS, and retraining (`test_audit_fixes.py`).
88. **[Medium] What is the main weakness of your counterfactual ground truth?**  
    Because it measures how much the *deployed model's* predictions change, it aligns directly with "consequential to this model" but partially favours model-aware scores ($D \times FI$, DIS) over model-blind scores.
89. **[Medium] How would you improve sensitivity $S$ in future work?**  
    Replace global random permutation with *shift-aware* sensitivity—measuring prediction change specifically under the observed reference-to-production shift rather than uniform shuffling across the whole range.
90. **[Easy] What are the top 3 features listed as "Not implemented" / future work?**  
    (1) Multivariate drift detection (MMD / classifier test), (2) dashboard threshold editing using the reserved `thresholds`/`threshold_history` tables, and (3) automated background scheduling / email alerts.
