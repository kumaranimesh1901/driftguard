# DriftGuard — Complete Explanation & Viva Preparation Pack

**Course:** Database Management Systems Laboratory (CD252IA) · 5th Semester B.E. CSE, RV College of Engineering  
**Purpose:** Everything you need to understand, present, and defend DriftGuard in your live demo and professor viva. Every claim cites the actual file and function in the repository, and every number comes directly from running the code, querying the live PostgreSQL/MongoDB databases, or reading `dis_prioritisation_results.json`.

## Index

| # | Document | What You Will Learn |
|---|---|---|
| 01 | [01_project_in_one_page.md](01_project_in_one_page.md) | 1-sentence pitch, 30s / 2m / 5m spoken scripts, the 7 questions DriftGuard answers, and the honest research answer |
| 02 | [02_architecture_and_data_flow.md](02_architecture_and_data_flow.md) | System architecture diagram, folder map, end-to-end CSV upload → analysis trace, and Argon2/JWT auth flow |
| 03 | [03_drift_detection_math.md](03_drift_detection_math.md) | PSI, KS, JSD, bootstrap noise floor, normalisation, D (max), severity, and ADWIN — formulas + hand-verified code examples |
| 04 | [04_model_shap_sensitivity_dis.md](04_model_shap_sensitivity_dis.md) | XGBoost credit model, real holdout metrics, TreeSHAP (FI), permutation sensitivity (S), and DIS = D × FI × S worked examples |
| 05 | [05_performance_decision_retraining.md](05_performance_decision_retraining.md) | Labelled performance bootstrap CI, 4 drift types, decision matrix, alerts, and champion–challenger retraining + rollback |
| 06 | [06_database_design.md](06_database_design.md) | **DBMS Deep Dive:** ER diagram, all 10 PostgreSQL tables, 1NF–3NF normalisation, ACID + Mongo compensation, 8 MongoDB collections, 10 SQL + 5 Mongo live queries |
| 07 | [07_api_and_dashboard.md](07_api_and_dashboard.md) | All 33 FastAPI endpoints, security controls, and all 10 Streamlit dashboard pages with demo talking points |
| 08 | [08_experiment_and_research_contribution.md](08_experiment_and_research_contribution.md) | Counterfactual experiment design, real results (D×FI vs DIS vs PSI), limitations, and what is novel vs existing |
| 09 | [09_demo_script.md](09_demo_script.md) | Pre-demo checklist, 10–12 minute click-by-click spoken demo script, and live failure recovery |
| 10 | [10_viva_questions_and_answers.md](10_viva_questions_and_answers.md) | 90 viva questions (Easy / Medium / Hard) with concise, honest, code-grounded answers |
| 11 | [11_glossary.md](11_glossary.md) | Plain-English definitions of every ML, statistics, security, and DBMS term with code references |
| All | [DriftGuard_Explained.md](DriftGuard_Explained.md) | Single merged document with Table of Contents, ready for PDF export |
