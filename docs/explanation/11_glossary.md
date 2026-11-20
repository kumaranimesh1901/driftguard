# 11 · Glossary of Terms

| Term | Plain-English Definition | Where It Appears in Code |
|---|---|---|
| **ACID** | Atomicity, Consistency, Isolation, Durability — guarantees that a PostgreSQL transaction succeeds completely or rolls back cleanly. | `app/services/monitoring/retraining.py → deploy()`, `rollback()` |
| **ADWIN** | Adaptive Windowing: streaming change detector that splits a sliding window into two sub-windows and flags a statistically significant mean shift. | `app/services/drift/adwin.py → adwin_stream()` |
| **Argon2** | Memory-hard password hashing algorithm resistant to GPU brute-force attacks. | `app/core/security.py → hash_password()` |
| **Associative Entity** | Bridge table resolving a Many-to-Many relationship (here `monitoring_jobs` ↔ `features`). | `db/init.sql → drift_summaries` |
| **Bootstrap** | Repeatedly resampling data with replacement to estimate uncertainty (noise floors or confidence intervals) without assuming a normal distribution. | `app/services/drift/calibration.py`, `model.py → performance_state()` |
| **Champion / Challenger** | Deployment pattern where the live model (`champion`) is compared against a newly trained candidate (`challenger`) on an independent test set before promotion. | `app/services/monitoring/retraining.py` |
| **CHECK Constraint** | Relational rule enforced by PostgreSQL rejecting invalid values (e.g., `severity IN ('low','moderate','high')`). | `db/init.sql` |
| **Concept Drift** | Change in the true input-to-target relationship $P(y \mid X)$ over time (same customer profile now defaults differently). | `app/services/monitoring/analysis.py → classify_drift_types()` |
| **Confusion Matrix** | 2×2 table of `[[TN, FP], [FN, TP]]` comparing predicted vs actual loan defaults. | `app/services/monitoring/model.py → classification_metrics()` |
| **CORS** | Cross-Origin Resource Sharing — browser security header restricting which web origins can call the API. | `app/main.py` |
| **Data Drift (Covariate Shift)** | Change in the distribution of input features $P(X)$ between training baseline and production. | `app/services/drift/*` |
| **Detector Agreement** | Summary of whether PSI, KS, and JSD agree (`agree-drift`, `agree-stable`, or `disagree`) relative to their noise floors. | `app/services/monitoring/analysis.py → compute_feature_drift()` |
| **DIS** | Drift Impact Score $= D \times FI \times S \in [0, 1]$, ranking drifted features by model impact. | `app/services/explain/dis.py → dis_variants()` |
| **F1-score** | Harmonic mean of precision and recall; used to choose the classification threshold on the training split. | `app/services/monitoring/model.py → select_threshold()` |
| **FI (Feature Importance)** | Normalised mean $|\text{SHAP}|$ attribution per feature on the production batch ($\in [0, 1]$). | `app/services/monitoring/model.py → shap_importance()` |
| **JSONB** | PostgreSQL's binary JSON column type supporting schema flexibility plus SQL key extraction (`->>`). | `db/init.sql → features.reference_stats`, `models.metrics` |
| **JSD** | Jensen–Shannon Divergence (base 2, $\in [0, 1]$): symmetric, bounded distribution distance derived from KL divergence. | `app/services/drift/jsd.py → jsd_from_counts()` |
| **JWT** | JSON Web Token — cryptographically signed (HS256) bearer token carrying user ID, role, and expiry. | `app/core/security.py → create_token()` |
| **KL Divergence** | Asymmetric information-theoretic measure of how one probability distribution diverges from another; basis of PSI and JSD. | `app/services/drift/psi.py`, `jsd.py` |
| **KS Test** | Kolmogorov–Smirnov two-sample test: maximum vertical gap between two empirical CDFs (`ks_stat`) plus its `p-value`. | `app/services/drift/ks.py → ks()` |
| **NDCG@3** | Normalised Discounted Cumulative Gain at top 3: ranking metric rewarding correct placement of the most consequential features at the top. | `scripts/experiments/dis_prioritisation.py → ndcg_at_k()` |
| **Noise Floor** | 95th-percentile drift value expected purely from random sampling under no drift; subtracted before scaling $D$. | `app/services/drift/calibration.py → numerical_noise_floors()` |
| **Normal Forms (1NF–3NF)** | Relational design rules eliminating repeating groups (1NF), partial dependencies (2NF), and transitive dependencies (3NF). | `db/init.sql` |
| **ORM** | Object–Relational Mapper (SQLAlchemy 2.0) mapping Python classes to PostgreSQL tables with parameterised queries. | `app/database/models.py` |
| **p-value** | Probability of observing a test statistic at least as extreme under the null hypothesis of no drift (not the same as effect size). | `app/services/drift/ks.py` |
| **Partial Unique Index** | Unique index applied only to rows matching a `WHERE` filter (e.g., `WHERE status='champion'`). | `db/init.sql → one_champion_per_dataset` |
| **Prediction Drift** | Shift in the model's predicted default probability distribution, detectable without outcome labels. | `app/services/monitoring/analysis.py → stream_monitoring()` |
| **PSI** | Population Stability Index: sum of $(p_k - q_k)\ln(p_k/q_k)$ across baseline quantile bins. | `app/services/drift/psi.py → psi()` |
| **RBAC** | Role-Based Access Control restricting endpoints to `reviewer`, `analyst`, or `administrator`. | `app/core/security.py → require_roles()` |
| **ROC-AUC** | Area Under the ROC Curve ($\in [0, 1]$): probability a random defaulter scores higher than a random repayer. | `app/services/monitoring/model.py → classification_metrics()` |
| **S (Model Sensitivity)** | Normalised mean $|\Delta \hat{p}|$ when permuting a feature 5 times within the production batch. | `app/services/monitoring/model.py → permutation_sensitivity()` |
| **SHAP** | SHapley Additive exPlanations: game-theoretic attribution of predictions to input features via `TreeExplainer`. | `app/services/monitoring/model.py → shap_importance()` |
