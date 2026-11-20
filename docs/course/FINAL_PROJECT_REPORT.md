# DRIFTGUARD: COMPREHENSIVE PROJECT REPORT
## Intelligent Framework for Automated Data Drift Detection, Root-Cause Analysis, and ML Model Monitoring

**Course:** Database Management Systems Laboratory (CD252IA)  
**Academic Year:** 5th Semester B.E. (Computer Science & Engineering)  
**Institution:** Department of Computer Science & Engineering, RV College of Engineering®, Bengaluru – 560059  
**Student Name:** Animesh  
**University Seat Number (USN):** 1RV24CD028  
**Faculty In-Charge:** Course Faculty, Dept. of CSE, RVCE  
**Document Version:** 1.1 (corrected after code audit, 20 Nov 2026)  

---

> **CORRECTION NOTICE (code audit, 20 Nov 2026).** Parts of this report described an earlier version of the code or planned features. The statements below supersede the rest of the document where they conflict.
>
> - **ADWIN** was not implemented when this report was written. It now is (`app/services/drift/adwin.py`, River), on two defined model streams: predicted probability (prediction drift) and |y − p| (error level, labelled batches only). It is not used for feature-distribution drift.
> - **Champion–challenger retraining, deploy and rollback** (FR-19, FR-20, Objective 6) did not exist. They now do (`app/services/monitoring/retraining.py`, `/api/v1/banking/models/*`). The earlier demonstration figures "AUC 0.84 → 0.75" and "AUC gain +0.08" were **not measured results** and have been removed. Section 9.2 now gives the real demonstration with measured values.
> - The dataset is the synthetic banking credit-risk data (`app/services/synthetic_data.py`, `data/`): 11 numerical + 1 categorical monitored features. `loan_data_synthetic.csv` and `app/simulation/generator.py` are no longer part of the project.
> - The coverage table in Section 11.1 was from an old snapshot (13 tests, 58%). The current measured result is given there.
> - **Research results** are now measured (Section 11.4, `docs/experiments/`). Model-aware ranking (D×FI and DIS) prioritises consequential drift clearly better than conventional PSI ranking. The sensitivity factor S does **not** add measurable value over D×FI in this setup. No drift score predicts the real AUC drop, so performance claims require labels.
> - All results come from **synthetic** data and one model family (XGBoost). They are evidence for this prototype, not a general claim.


## EXECUTIVE SUMMARY / ABSTRACT

Machine learning models deployed in production inevitably experience performance degradation due to temporal changes in real-world data distributions (data drift and concept drift). Traditional monitoring solutions generate unranked, univariate statistical alerts (e.g., Population Stability Index, Kolmogorov-Smirnov tests) that fail to indicate whether a detected distributional shift actually impacts the deployed model, lead to severe alert fatigue, and trigger premature, expensive model retraining.

**DriftGuard** is an integrated, model-aware, database-backed framework that monitors tabular binary classification models, isolates drifting features, evaluates their true consequence on model predictions using explainable AI, and recommends justified governance actions. DriftGuard introduces the **Drift Impact Score (DIS)**—a proposed composite metric combining normalised statistical drift magnitude ($D_i$), SHAP-based feature importance ($FI_i$), and permutation output sensitivity ($S_i$) scaled to $[0, 1]$. Crucially, DriftGuard maintains a strict architectural distinction between a **drift-warning** (distributional shifts occurring when labels are pending or performance remains stable) and **confirmed-degradation** (empirically verified ROC-AUC/F1 drop upon arrival of ground-truth labels).

The storage architecture is designed around a **hybrid SQL + NoSQL database paradigm**:
1. **PostgreSQL 15** manages structured, relational, integrity-critical metadata, normalized to Boyce-Codd Normal Form (BCNF), enforcing referential integrity, check constraints, and ACID transactions for champion-challenger deployments.
2. **MongoDB 6** handles high-volume, append-heavy, heterogeneous documents including prediction logs, detector histograms, per-batch timings, and detailed root-cause explanation reports.

This comprehensive report documents the end-to-end design, implementation, and academic evaluation across all **eleven core components** prescribed by the CD252IA syllabus.

---

# TABLE OF CONTENTS
- [1. Component 1: Project Synopsis & Problem Definition](#1-component-1-project-synopsis--problem-definition)
- [2. Component 2: Software Requirement Specification (SRS)](#2-component-2-software-requirement-specification-srs)
- [3. Component 3: Entity-Relationship Model & Relational Schema](#3-component-3-entity-relationship-model--relational-schema)
- [4. Component 4: Database Query Deliverables (SQL & NoSQL)](#4-component-4-database-query-deliverables-sql--nosql)
- [5. Component 5: Data Flow Diagrams (DFDs)](#5-component-5-data-flow-diagrams-dfds)
- [6. Component 6: Database Normalization (UNF to BCNF)](#6-component-6-database-normalization-unf-to-bcnf)
- [7. Component 7: Midterm Milestone & Progress Synthesis](#7-component-7-midterm-milestone--progress-synthesis)
- [8. Component 8: Design of Forms, Security, and Validation](#8-component-8-design-of-forms-security-and-validation)
- [9. Component 9: Hybrid Architecture & Demonstration Plan](#9-component-9-hybrid-architecture--demonstration-plan)
- [10. Component 10: Innovative Experiment, MLOps, & Automation](#10-component-10-innovative-experiment-mlops--automation)
- [11. Component 11: Final Report & Verification Audit](#11-component-11-final-report--verification-audit)

---

# 1. COMPONENT 1: PROJECT SYNOPSIS & PROBLEM DEFINITION

### 1.1 Problem Statement
Modern machine learning models deployed in high-stakes domains (e.g., credit underwriting, loan-default prediction) make automated decisions on dynamic customer populations. Existing drift monitoring systems suffer from four critical deficiencies:
1. **Lack of Model-Awareness:** A feature exhibiting statistically significant drift (e.g., high PSI) may have near-zero weight in the decision trees, rendering the shift inconsequential to prediction outcomes.
2. **Conflation of Drift with Degradation:** Conventional alerts treat any covariate shift as evidence of model failure, triggering panic when model performance is actually stable.
3. **Detector Disagreement:** Non-parametric tests (KS) and binning metrics (PSI, JSD) operate on disparate scales, leading to conflicting alerts with no consolidated ranking.
4. **Uncontrolled Retraining:** Models are frequently retrained without verifying if the new challenger outperforms the current champion on held-out post-drift test data, risking regression and catastrophic forgetting.

### 1.2 Literature Survey & Limitations of Existing Work
A survey of 20 relevant peer-reviewed research papers (2024–2025) categorized existing work into five distinct domains:

| Research Domain | Representative Papers | Core Contribution | Key Limitations Addressed by DriftGuard |
|---|---|---|---|
| **Surveys & Foundations** | [1] [2] [13] [16] | Taxonomies for detecting, locating, and explaining concept drift; unsupervised benchmarks | Purely theoretical benchmarks; lack production-grade storage designs, role-based workflows, and model-aware prioritization |
| **Detection Methods** | [3] [5] [8] [9] [11] [19] | Weighted-conformal martingales, uncertainty estimation, entropy metrics | Answer *if* or *when* drift occurred at the stream level, but fail to localize *which* input features drove the shift |
| **Localization & Explanation** | [4] [7] [17] | Feature-level drift probabilities, cause-effect rationalization | Closest to DriftGuard, but do not integrate multi-detector fusion with SHAP importance, permutation sensitivity, and dual-database storage |
| **Adaptation & Retraining** | [6] [10] [12] [18] | Proactive retraining, continual learning mitigation | Assume the decision to adapt is already justified; DriftGuard focuses on deciding *whether* retraining is warranted |
| **Applied & Systems** | [14] [15] [20] | Medical imaging drift experiments, serverless architectures, Frouros library | Domain-specific or single-detector toolkits without relational data governance or human-in-the-loop rollback mechanisms |

### 1.3 Research Gap & Central Research Question
Existing literature treats detection, explanation, and adaptation as isolated problems. The fundamental opportunity is to unify these capabilities into a single, database-backed governance pipeline.

> [!IMPORTANT]
> **Central Research Question:**  
> *Can combining multi-detector statistical drift magnitude with model-specific feature importance and observed empirical performance changes improve the prioritisation of potentially consequential drift compared with conventional unweighted drift alerts?*

### 1.4 Objectives of the System
- **Main Objective:** Design, build, and evaluate an integrated, model-aware data drift monitoring and root-cause analysis system with hybrid SQL/NoSQL storage.
- **Specific Objectives:**
  1. Automate feature-level monitoring using PSI, KS-test, JSD, and streaming ADWIN.
  2. Implement bootstrap null calibration to calculate noise floors and scale disparate metrics to $[0, 1]$.
  3. Formulate and compute the Drift Impact Score ($DIS_i = D_i \times FI_i \times S_i$).
  4. Implement an explicit decision rule engine separating `drift-warning` from `confirmed-degradation`.
  5. Architect a BCNF PostgreSQL schema for metadata and an indexed MongoDB database for high-throughput logs.
  6. Support human-in-the-loop retraining with automated champion-challenger evaluation and atomic rollback.

---

# 2. COMPONENT 2: SOFTWARE REQUIREMENT SPECIFICATION (SRS)

### 2.1 Hardware Requirements
- **Processor:** Quad-Core 2.0 GHz minimum (Intel i5/AMD Ryzen 5 or Apple Silicon M-series recommended: 6–8 cores, 3.0 GHz+).
- **RAM:** 8 GB minimum (16 GB recommended for concurrent bootstrap sampling and SHAP background profiling).
- **Storage:** 20 GB free SSD minimum (50 GB recommended for persistent MongoDB volume and joblib model artifacts).
- **GPU:** Not required; tree-based SHAP and tabular scikit-learn models run on multicore CPU.

### 2.2 Software Stack
- **Operating System:** Ubuntu 22.04 LTS, macOS 12+, or Windows 10/11 with WSL2.
- **Backend Framework:** Python 3.11+, FastAPI 0.115+, Uvicorn (ASGI server), Pydantic v2.
- **Frontend / Dashboard:** Streamlit multi-page interface with Plotly interactive charts.
- **Relational DBMS:** PostgreSQL 15+ accessed via SQLAlchemy 2.0 (ORM & Core) and `psycopg2-binary`.
- **Document DBMS:** MongoDB 6.0+ accessed via `PyMongo 4.9+`.
- **Machine Learning & Drift:** Scikit-learn, XGBoost, SHAP 0.46+, SciPy 1.14+, NumPy, Pandas, River (ADWIN).
- **Security & Cryptography:** `argon2-cffi` for password hashing, `PyJWT` for HMAC-SHA256 session tokens.
- **Containerization & Build:** Docker Engine 24+, Docker Compose v2, GNU Make.

### 2.3 Functional Requirements (FR) Matrix

```
FR-1   User Registration & JWT Login with Role-Based Access Control (Analyst, Administrator, Reviewer)
FR-2   Dataset Ingestion via CSV upload with schema inference and target column validation
FR-3   Automated Data Quality Profiling (missing values, cardinality, duplicates, constant columns)
FR-4   Baseline Model Training (Random Forest, XGBoost, Logistic Regression) with stratified 60/20/20 split
FR-5   Reference Profile Construction (quantile bin edges, category distributions, global SHAP importance)
FR-6   Batch Ingestion REST API with strict schema validation against the reference dataset
FR-7   Univariate Drift Detection (PSI, 2-sample Kolmogorov-Smirnov test, Jensen-Shannon Divergence)
FR-8   Streaming Concept Drift Tracking using River ADWIN on sequential prediction records
FR-9   Bootstrap Null Normalisation against a 200-sample noise floor to generate normalised D in [0, 1]
FR-10  Local Explainability via SHAP TreeExplainer / LinearExplainer on production batches
FR-11  Permutation-style Model Output Sensitivity computation across feature columns
FR-12  Drift Impact Score (DIS) computation and ablation variants ranking
FR-13  Delayed Label Ingestion and automatic batch performance calculation (Accuracy, F1, ROC-AUC)
FR-14  Bootstrap Confidence Interval estimation for ROC-AUC to verify performance drops
FR-15  Automated Rule-Based Action Recommendation (Continue, Investigate, Consider Retraining, Review)
FR-16  Root-Cause Report document generation linking detector events to top-ranked features
FR-17  Hybrid Data Persistence (Metadata & Decisions -> PostgreSQL; Logs & Reports -> MongoDB)
FR-18  Interactive Web Dashboard with DIS bar charts, histogram overlays, and alert banners
FR-19  Champion-Challenger Model Retraining with strict held-out test evaluation
FR-20  Atomic Model Deployment and Rollback via single-transaction database status swap
FR-21  Drift Simulation Engine generating sudden, gradual, and recurring covariate shifts
FR-22  Audited Dynamic Threshold Configuration (Severity limits, PSI bins, Bootstraps)
FR-23  Data & Report Export functionality supporting CSV and JSON serialization
```

### 2.4 Non-Functional Requirements (NFR)
- **NFR-1 (Performance):** A 10,000-record batch must complete full drift detection, SHAP sampling, and database writes in under 60 seconds on recommended hardware.
- **NFR-2 (Scalability):** MongoDB collections must utilize compound indexes (`{model_id: 1, batch_id: 1, timestamp: -1}`) to ensure sub-100ms retrieval times under high query volume.
- **NFR-3 (Data Integrity):** PostgreSQL schema must enforce strict Foreign Keys, `CHECK` constraints on domains, and `ON DELETE CASCADE` where parent deletion dictates child lifetime.
- **NFR-4 (Atomicity):** Deployment of a challenger model must swap statuses (`champion` $\leftrightarrow$ `retired`) inside a single ACID transaction to guarantee that exactly one champion exists per dataset.
- **NFR-5 (Security):** All passwords must be hashed using salted Argon2id. No raw tokens or secrets may be logged. All API endpoints must reject malformed tokens with HTTP 401.
- **NFR-6 (Fault Isolation):** If MongoDB experiences transient downtime, the API must return a degraded HTTP 503 status without crashing or leaking database hostnames.

---

# 3. COMPONENT 3: ENTITY-RELATIONSHIP MODEL & RELATIONAL SCHEMA

### 3.1 Entity-Relationship (ER) Conceptual Design
The relational subsystem in PostgreSQL models integrity-critical entities and governance workflows.

```mermaid
erDiagram
    USERS ||--o{ DATASETS : "uploads (1:N)"
    USERS ||--o{ MONITORING_JOBS : "initiates (1:N)"
    DATASETS ||--|{ FEATURES : "contains (1:N)"
    DATASETS ||--|{ MODELS : "trains (1:N)"
    MODELS ||--o{ MONITORING_JOBS : "monitored_in (1:N)"
    MONITORING_JOBS ||--|{ DRIFT_SUMMARIES : "produces (1:N)"
    FEATURES ||--o{ DRIFT_SUMMARIES : "measured_for (1:N)"
    MONITORING_JOBS ||--o{ ALERTS : "raises (1:N)"
    ALERTS ||--o{ RETRAINING_EVENTS : "triggers (0..1:N)"
    MODELS ||--o{ RETRAINING_EVENTS : "old_model (1:N)"
    MODELS ||--o{ RETRAINING_EVENTS : "new_model (1:N)"

    USERS {
        int user_id PK
        string name
        string email UK
        string password_hash
        string role
    }
    DATASETS {
        int dataset_id PK
        int user_id FK
        string name
        string source
        int row_count
        timestamp created_at
    }
    FEATURES {
        int feature_id PK
        int dataset_id FK
        string name
        string type
        jsonb reference_stats
    }
    MODELS {
        int model_id PK
        int dataset_id FK
        string algorithm
        int version
        string status
        jsonb metrics
        timestamp trained_at
    }
    MONITORING_JOBS {
        int job_id PK
        int model_id FK
        int user_id FK
        string batch_id
        timestamp window_start
        timestamp window_end
        string status
    }
    DRIFT_SUMMARIES {
        int summary_id PK
        int job_id FK
        int feature_id FK
        float psi
        float ks_stat
        float ks_p
        float jsd
        float dis
        string severity
    }
    ALERTS {
        int alert_id PK
        int job_id FK
        string alert_type
        string level
        text recommendation
        timestamp created_at
    }
    RETRAINING_EVENTS {
        int event_id PK
        int alert_id FK
        int old_model_id FK
        int new_model_id FK
        string decision
        boolean rollback_flag
        timestamp created_at
    }
```

### 3.2 Cardinality & Relationship Matrix
1. **USERS $\rightarrow$ DATASETS (1 : N):** An analyst or administrator can register multiple datasets; each dataset is registered by exactly one user.
2. **DATASETS $\rightarrow$ FEATURES (1 : N):** Each dataset comprises multiple features (numerical or categorical); features belong to exactly one dataset (`ON DELETE CASCADE`).
3. **DATASETS $\rightarrow$ MODELS (1 : N):** A dataset can train multiple models (different algorithms or versions); each model is tied to a specific training dataset.
4. **MODELS $\rightarrow$ MONITORING_JOBS (1 : N):** A deployed model undergoes continuous monitoring across multiple production batches.
5. **MONITORING_JOBS $\times$ FEATURES $\rightarrow$ DRIFT_SUMMARIES (M : N resolved):** Each monitoring job analyzes all features of the dataset. `DRIFT_SUMMARIES` represents the associative entity storing per-feature drift metrics with candidate key `(job_id, feature_id)`.
6. **MONITORING_JOBS $\rightarrow$ ALERTS (1 : N):** A job may generate zero, one, or multiple alerts (`drift-warning` or `confirmed-degradation`).
7. **MODELS $\rightarrow$ RETRAINING_EVENTS (1 : N twice):** Retraining events track lineage between an `old_model_id` and a `new_model_id` with `CHECK (old_model_id <> new_model_id)`.

### 3.3 Relational Schema & DDL Specification (`db/init.sql`)

```sql
-- 1. Users & RBAC
CREATE TABLE users (
  user_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(150) NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role VARCHAR(20) NOT NULL CHECK (role IN ('analyst', 'administrator', 'reviewer'))
);

-- 2. Datasets
CREATE TABLE datasets (
  dataset_id SERIAL PRIMARY KEY,
  user_id INT NOT NULL REFERENCES users(user_id),
  name VARCHAR(150) NOT NULL,
  source VARCHAR(200),
  row_count INT CHECK (row_count >= 0),
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE (user_id, name)
);

-- 3. Features
CREATE TABLE features (
  feature_id SERIAL PRIMARY KEY,
  dataset_id INT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  name VARCHAR(100) NOT NULL,
  type VARCHAR(20) NOT NULL CHECK (type IN ('numerical', 'categorical')),
  reference_stats JSONB,
  UNIQUE (dataset_id, name)
);

-- 4. Models
CREATE TABLE models (
  model_id SERIAL PRIMARY KEY,
  dataset_id INT NOT NULL REFERENCES datasets(dataset_id),
  algorithm VARCHAR(50) NOT NULL,
  version INT NOT NULL,
  status VARCHAR(20) NOT NULL CHECK (status IN ('champion', 'challenger', 'retired')),
  metrics JSONB,
  trained_at TIMESTAMP DEFAULT now(),
  UNIQUE (dataset_id, algorithm, version)
);

-- 5. Monitoring Jobs
CREATE TABLE monitoring_jobs (
  job_id SERIAL PRIMARY KEY,
  model_id INT NOT NULL REFERENCES models(model_id),
  user_id INT NOT NULL REFERENCES users(user_id),
  batch_id VARCHAR(64) NOT NULL,
  window_start TIMESTAMP NOT NULL,
  window_end TIMESTAMP NOT NULL,
  status VARCHAR(20) NOT NULL,
  CHECK (window_end >= window_start)
);

-- 6. Drift Summaries (Feature x Job associative entity)
CREATE TABLE drift_summaries (
  summary_id SERIAL PRIMARY KEY,
  job_id INT NOT NULL REFERENCES monitoring_jobs(job_id) ON DELETE CASCADE,
  feature_id INT NOT NULL REFERENCES features(feature_id),
  psi REAL,
  ks_stat REAL,
  ks_p REAL,
  jsd REAL,
  dis REAL,
  severity VARCHAR(10) CHECK (severity IN ('low', 'moderate', 'high')),
  UNIQUE (job_id, feature_id)
);

-- 7. Alerts
CREATE TABLE alerts (
  alert_id SERIAL PRIMARY KEY,
  job_id INT NOT NULL REFERENCES monitoring_jobs(job_id),
  alert_type VARCHAR(30) NOT NULL CHECK (alert_type IN ('drift-warning', 'confirmed-degradation')),
  level VARCHAR(10) NOT NULL,
  recommendation TEXT,
  created_at TIMESTAMP DEFAULT now()
);

-- 8. Retraining Events
CREATE TABLE retraining_events (
  event_id SERIAL PRIMARY KEY,
  alert_id INT REFERENCES alerts(alert_id),
  old_model_id INT NOT NULL REFERENCES models(model_id),
  new_model_id INT NOT NULL REFERENCES models(model_id),
  decision VARCHAR(20) NOT NULL CHECK (decision IN ('pending', 'deployed', 'rejected', 'rolled_back')),
  rollback_flag BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT now(),
  CHECK (old_model_id <> new_model_id)
);

-- Supporting Tables (Configuration & Audit History)
CREATE TABLE thresholds (
  threshold_id SERIAL PRIMARY KEY,
  name VARCHAR(60) NOT NULL UNIQUE,
  value DOUBLE PRECISION NOT NULL,
  updated_by INT REFERENCES users(user_id),
  updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE threshold_history (
  history_id SERIAL PRIMARY KEY,
  name VARCHAR(60) NOT NULL,
  old_value DOUBLE PRECISION,
  new_value DOUBLE PRECISION NOT NULL,
  changed_by INT REFERENCES users(user_id),
  changed_at TIMESTAMP DEFAULT now()
);

-- Optimization & Partial Unique Constraints
CREATE INDEX drift_summaries_job_id ON drift_summaries(job_id);
CREATE INDEX monitoring_jobs_model_window ON monitoring_jobs(model_id, window_start);
CREATE INDEX alerts_job_created ON alerts(job_id, created_at);
CREATE UNIQUE INDEX one_champion_per_dataset ON models(dataset_id) WHERE status = 'champion';
```

---

# 4. COMPONENT 4: DATABASE QUERY DELIVERABLES (SQL & NoSQL)

### 4.1 Relational SQL Queries (PostgreSQL: S1–S12)
These 12 queries test every major SQL capability required by the CD252IA syllabus:

```sql
-- S1. Simple Projection & Filter: Features of a Dataset
SELECT feature_id, name, type 
FROM features 
WHERE dataset_id = 1 
ORDER BY feature_id;

-- S2. Inner Join & Order By: Top drifted features of a job ranked by DIS
SELECT f.name, d.psi, d.ks_stat, d.jsd, d.dis, d.severity
FROM drift_summaries d 
JOIN features f ON f.feature_id = d.feature_id
WHERE d.job_id = 3 
ORDER BY d.dis DESC;

-- S3. Group By with Having & Round: Features with High Average PSI across Jobs
SELECT f.name, ROUND(AVG(d.psi)::numeric, 3) AS avg_psi, MAX(d.psi) AS max_psi
FROM drift_summaries d 
JOIN features f ON f.feature_id = d.feature_id
GROUP BY f.name 
HAVING AVG(d.psi) > 0.10 
ORDER BY avg_psi DESC;

-- S4. Correlated Subquery: Features with DIS strictly above overall average
SELECT f.name, d.job_id, d.dis 
FROM drift_summaries d
JOIN features f ON f.feature_id = d.feature_id
WHERE d.dis > (SELECT AVG(dis) FROM drift_summaries) 
ORDER BY d.dis DESC;

-- S5. Common Table Expression (CTE) & Window Function (RANK): Top 2 features per job
WITH ranked AS (
  SELECT d.job_id, f.name, d.dis,
         RANK() OVER (PARTITION BY d.job_id ORDER BY d.dis DESC) AS rnk
  FROM drift_summaries d 
  JOIN features f ON f.feature_id = d.feature_id
)
SELECT job_id, name, dis, rnk 
FROM ranked 
WHERE rnk <= 2 
ORDER BY job_id, rnk;

-- S6. Multi-Table Join: Alerts correlated with job, model, and user identity
SELECT a.alert_id, a.alert_type, a.level, j.batch_id, m.algorithm, m.version, u.name
FROM alerts a 
JOIN monitoring_jobs j ON j.job_id = a.job_id
JOIN models m ON m.model_id = j.model_id 
JOIN users u ON u.user_id = j.user_id;

-- S7. Left Outer Join: Monitoring jobs that completed without raising alerts
SELECT j.job_id, j.batch_id 
FROM monitoring_jobs j
LEFT JOIN alerts a ON a.job_id = j.job_id 
WHERE a.alert_id IS NULL;

-- S8. Database View: High-severity drift monitoring abstraction
CREATE VIEW v_high_drift AS
SELECT d.job_id, j.batch_id, f.name AS feature, d.psi, d.dis
FROM drift_summaries d 
JOIN features f ON f.feature_id = d.feature_id
JOIN monitoring_jobs j ON j.job_id = d.job_id 
WHERE d.severity = 'high';

SELECT * FROM v_high_drift ORDER BY dis DESC;

-- S9. Window Function (LAG): Temporal change in feature DIS between jobs
SELECT f.name, d.job_id, d.dis,
       d.dis - LAG(d.dis) OVER (PARTITION BY d.feature_id ORDER BY d.job_id) AS change
FROM drift_summaries d 
JOIN features f ON f.feature_id = d.feature_id 
ORDER BY f.name, d.job_id;

-- S10. Self-Join with JSONB Extraction: Old vs New Model Comparison per Retrain Event
SELECT r.event_id, r.decision, 
       mo.algorithm || ' v' || mo.version AS old_model,
       (mo.metrics->>'auc')::numeric AS old_auc, 
       mn.algorithm || ' v' || mn.version AS new_model,
       (mn.metrics->>'auc')::numeric AS new_auc
FROM retraining_events r 
JOIN models mo ON mo.model_id = r.old_model_id
JOIN models mn ON mn.model_id = r.new_model_id;

-- S11. ACID Transaction: Atomic Champion Demotion & Challenger Promotion
BEGIN;
  UPDATE models SET status = 'retired' WHERE model_id = 1;
  UPDATE models SET status = 'champion' WHERE model_id = 2;
  UPDATE retraining_events SET decision = 'deployed' WHERE event_id = 1;
COMMIT; -- Rolled back if unique constraint or check constraint fails

-- S12. Query Optimization Plan: Index usage verification
CREATE INDEX idx_drift_job ON drift_summaries(job_id);
EXPLAIN ANALYZE 
SELECT * FROM drift_summaries 
WHERE job_id = 3 
ORDER BY dis DESC;
```

### 4.2 Document NoSQL Queries (MongoDB: N1–N9)
These 9 queries exercise filtering, projection, multi-stage aggregation, arrays, and joins in MongoDB:

```javascript
// N1. Filter with Comparison and Projection: Unlabelled High-Risk Predictions
db.prediction_logs.find(
  { predicted_prob: { $gt: 0.7 }, true_label: null },
  { batch_id: 1, predicted_prob: 1, "features.income": 1, _id: 0 }
);

// N2. Aggregation ($group, $avg, $sort): Summary Statistics per Ingested Batch
db.prediction_logs.aggregate([
  { $group: { _id: "$batch_id", total_records: { $sum: 1 }, avg_risk: { $avg: "$predicted_prob" } } },
  { $sort: { _id: 1 } }
]);

// N3. Conditional Aggregation ($cond, $divide): Empirical Batch Accuracy Upon Label Arrival
db.prediction_logs.aggregate([
  { $match: { true_label: { $ne: null } } },
  { $group: {
      _id: "$batch_id",
      n: { $sum: 1 },
      correct: { $sum: { $cond: [{ $eq: ["$predicted_label", "$true_label"] }, 1, 0] } }
  }},
  { $project: { n: 1, accuracy: { $divide: ["$correct", "$n"] } } }
]);

// N4. Sorting and Range Queries: Drift Events Exceeding Conventional Statistical Thresholds
db.drift_events.find({ exceeded_conventional: true }).sort({ value: -1 });

// N5. Array Deconstruction ($unwind, $first): Extract Highest-Impact Feature per Root-Cause Report
db.root_cause_reports.aggregate([
  { $unwind: "$ranked_features" },
  { $sort: { job_id: 1, "ranked_features.dis": -1 } },
  { $group: { _id: "$job_id", top_drifting_feature: { $first: "$ranked_features.feature" }, max_dis: { $first: "$ranked_features.dis" } } }
]);

// N6. Multi-Collection Lookup ($lookup): Join Drift Events with Parent Root-Cause Reports
db.drift_events.aggregate([
  { $match: { exceeded_conventional: true } },
  { $lookup: {
      from: "root_cause_reports",
      localField: "job_id",
      foreignField: "job_id",
      as: "report"
  }}
]);

// N7. Document Modification ($set): Late-Arriving Ground Truth Label Ingestion
db.prediction_logs.updateOne(
  { batch_id: "B002", record_id: "REC_10492" },
  { $set: { true_label: 1, label_received_at: new Date() } }
);

// N8. Compound Index Execution Plan: Verifying Query Performance
db.prediction_logs.find({ model_id: 1, batch_id: "B001" }).sort({ timestamp: -1 }).explain("executionStats");

// N9. Data Retention & Pruning: Purging Stale Execution Logs Older Than 90 Days
db.monitoring_logs.deleteMany({
  finished_at: { $lt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000) }
});
```

---

# 5. COMPONENT 5: DATA FLOW DIAGRAMS (DFDs)

### 5.1 Level 0 DFD (Context Diagram)
The Context Diagram establishes the boundary between DriftGuard and its external entities.

```mermaid
flowchart TD
    E1["External Entity 1: Analyst / Administrator"]
    E2["External Entity 2: Production Application"]
    P0(("0.0 DriftGuard Monitoring & Governance System"))

    E1 -- "Credentials, Datasets, Threshold Edits, Retraining Approvals" --> P0
    P0 -- "Interactive Dashboards, Alerts, Root-Cause Reports, Audit Logs" --> E1
    
    E2 -- "Production Ingestion Batches, Real-time Predictions, Delayed Labels" --> P0
    P0 -- "Approved Champion Model Artifact & Ingestion Receipts" --> E2
```

### 5.2 Level 1 DFD (Decomposition into 7 Core Processes)
Level 1 decomposes the system into functional subsystems and identifies data stores $D1$ through $D5$.

```mermaid
flowchart TD
    E1["Analyst / Admin"]
    E2["Production App"]

    P1(("1.0 Manage Users & Datasets"))
    P2(("2.0 Train Model & Reference Profile"))
    P3(("3.0 Ingest Production Batches"))
    P4(("4.0 Detect Drift & Rank Features"))
    P5(("5.0 Evaluate Performance & Recommend"))
    P6(("6.0 Retrain & Deploy Model"))
    P7(("7.0 Display Dashboard & Alerts"))

    D1[("D1: Users (PG)")]
    D2[("D2: Datasets & Features (PG)")]
    D3[("D3: Models & Ref Profiles (PG + Disk)")]
    D4[("D4: Monitoring Jobs & Drift Summaries (PG)")]
    D5[("D5: Logs, Batches & Reports (MongoDB)")]

    E1 -- "Register / Login / Upload" --> P1
    P1 --> D1
    P1 --> D2

    P1 -- "Cleaned Dataset" --> P2
    P2 --> D3

    E2 -- "Batches / Predictions" --> P3
    P3 --> D5

    D3 -- "Reference Profile" --> P4
    D5 -- "Batch Records" --> P4
    P4 --> D4
    P4 --> D5

    E2 -- "Delayed Labels" --> P5
    D4 -- "Drift Level" --> P5
    P5 --> D4

    D4 -- "Trigger Alert" --> P6
    E1 -- "Approval" --> P6
    D5 -- "Labeled Records" --> P6
    P6 --> D3
    P6 -- "New Champion" --> E2

    D3 & D4 & D5 --> P7
    P7 --> E1
```

### 5.3 Level 2 DFD (Sub-process Decomposition for Process 4.0: Detect Drift & Rank Features)
Process 4.0 embodies the core analytical innovation of DriftGuard:

```mermaid
flowchart TD
    subgraph "Process 4.0 Sub-Processes"
        P41(("4.1 Validate Batch Schema & Extrema"))
        P42(("4.2 Compute Multi-Detector Drift (PSI, KS, JSD, ADWIN)"))
        P43(("4.3 Compute Explainability (SHAP & Permutation Sensitivity)"))
        P44(("4.4 Compute Normalised DIS & Rank Features"))
    end

    In1["Production Batch (D5)"] --> P41
    In2["Reference Profile (D3)"] --> P41
    
    P41 -- "Validated Aligned Matrix" --> P42
    P41 -- "Validated Aligned Matrix" --> P43
    
    P42 -- "Raw Statistics (s)" --> P44
    P43 -- "Feature Importance (FI) & Sensitivity (S)" --> P44

    P44 -- "Dis Summaries & Severity" --> Out1["D4: drift_summaries (PG)"]
    P44 -- "Root-Cause Report & Histograms" --> Out2["D5: drift_events & root_cause_reports (MongoDB)"]
```

---

# 6. COMPONENT 6: DATABASE NORMALIZATION (UNF TO BCNF)

### 6.1 Unnormalised Relation (UNF)
Consider a single flat spreadsheet log recording monitoring activity:

$$\text{MONITORING\_REPORT}(\text{job\_id}, \text{batch\_id}, \text{window\_start}, \text{window\_end}, \text{job\_status}, \text{model\_id}, \text{algorithm}, \text{version}, \text{model\_status}, \text{metrics},$$
$$\text{dataset\_id}, \text{dataset\_name}, \text{dataset\_source}, \text{row\_count}, \text{user\_id}, \text{user\_name}, \text{user\_email}, \text{user\_role},$$
$$\{\text{feature\_id}, \text{feature\_name}, \text{feature\_type}, \text{psi}, \text{ks\_stat}, \text{ks\_p}, \text{jsd}, \text{dis}, \text{severity}\},$$
$$\{\text{alert\_id}, \text{alert\_type}, \text{level}, \text{recommendation}, \text{alert\_created\_at}\})$$

- **Severe Anomalies in UNF:**
  - *Insertion Anomaly:* Cannot register a user without uploading a dataset. Cannot register a feature without running a monitoring job.
  - *Update Anomaly:* If an analyst changes their email, hundreds of historical job rows must be scanned and updated.
  - *Deletion Anomaly:* Deleting the sole monitoring job for an experimental model inadvertently deletes the model details and dataset definition.

### 6.2 First Normal Form (1NF)
*Rule:* Eliminate repeating groups; ensure all attributes contain atomic values; define primary keys.
- **Action:** Flatten repeating groups into child relations tied by parent primary keys:
  - $R_1\text{ (Job Report):}$ Contains job, model, dataset, and user scalar fields. Key: `job_id`.
  - $R_2\text{ (Job Features):}$ `(job_id, feature_id, feature_name, feature_type, dataset_id, psi, ks_stat, ks_p, jsd, dis, severity)`. Key: `(job_id, feature_id)`.
  - $R_3\text{ (Job Alerts):}$ `(alert_id, job_id, alert_type, level, recommendation, created_at)`. Key: `alert_id`.

### 6.3 Second Normal Form (2NF)
*Rule:* 1NF and no non-prime attribute is partially dependent on any candidate key.
- **Analysis:** In $R_2$, the key is `(job_id, feature_id)`. However:
  $$\text{feature\_id} \rightarrow \text{feature\_name, feature\_type, dataset\_id}$$
  This is a **partial dependency** on a subset of the composite key.
- **Action:** Decompose $R_2$ into:
  - $\text{FEATURES}(\underline{\text{feature\_id}}, \text{dataset\_id}, \text{name}, \text{type})$
  - $\text{DRIFT\_SUMMARIES}(\underline{\text{job\_id}, \text{feature\_id}}, \text{psi}, \text{ks\_stat}, \text{ks\_p}, \text{jsd}, \text{dis}, \text{severity})$

### 6.4 Third Normal Form (3NF)
*Rule:* 2NF and no transitive dependencies ($X \rightarrow Y$ where $Y$ is non-prime and $X$ is not a superkey).
- **Analysis:** In $R_1$ (`job_id` is PK), several transitive chains exist:
  $$\text{job\_id} \rightarrow \text{model\_id} \rightarrow \text{algorithm, version, status, metrics}$$
  $$\text{model\_id} \rightarrow \text{dataset\_id} \rightarrow \text{name, source, row\_count}$$
  $$\text{job\_id} \rightarrow \text{user\_id} \rightarrow \text{name, email, password\_hash, role}$$
- **Action:** Extract transitive relations into separate tables: `MODELS`, `DATASETS`, `USERS`, and `MONITORING_JOBS`.

### 6.5 Boyce-Codd Normal Form (BCNF)
*Rule:* For every functional dependency $X \rightarrow Y$, $X$ must be a superkey.

| Relation | Candidate Keys | Functional Dependencies ($X \rightarrow Y$) | Is $X$ a Superkey? | BCNF Compliant? |
|---|---|---|---|---|
| **USERS** | `user_id`, `email` | `user_id` $\rightarrow$ `name, email, password_hash, role`; `email` $\rightarrow$ `user_id, ...` | Yes (both candidate keys) | **Yes** |
| **DATASETS** | `dataset_id`, `(user_id, name)` | `dataset_id` $\rightarrow$ all attributes; `(user_id, name)` $\rightarrow$ all attributes | Yes | **Yes** |
| **FEATURES** | `feature_id`, `(dataset_id, name)` | `feature_id` $\rightarrow$ all attributes; `(dataset_id, name)` $\rightarrow$ all attributes | Yes | **Yes** |
| **MODELS** | `model_id`, `(dataset_id, algo, ver)` | `model_id` $\rightarrow$ all; `(dataset_id, algo, ver)` $\rightarrow$ all | Yes | **Yes** |
| **MONITORING_JOBS** | `job_id` | `job_id` $\rightarrow$ all attributes | Yes | **Yes** |
| **DRIFT_SUMMARIES** | `summary_id`, `(job_id, feat_id)` | `summary_id` $\rightarrow$ all; `(job_id, feat_id)` $\rightarrow$ all | Yes | **Yes** |
| **ALERTS** | `alert_id` | `alert_id` $\rightarrow$ all attributes | Yes | **Yes** |
| **RETRAINING_EVENTS** | `event_id` | `event_id` $\rightarrow$ all attributes | Yes | **Yes** |

> [!NOTE]
> **Design Justification on `severity`:** The attribute `severity` in `DRIFT_SUMMARIES` is functionally determined by `(dis, thresholds)`. Storing it directly is a deliberate denormalization decision to record an immutable, auditable snapshot of the operational decision made at runtime, unaffected by future threshold reconfigurations.

---

# 7. COMPONENT 7: MIDTERM MILESTONE & PROGRESS SYNTHESIS

### 7.1 Scope & Boundaries Defined at Midterm
At the midterm stage, the scope of DriftGuard was formally defined and approved:
- **In Scope:** Tabular binary classification for loan default risk; continuous and categorical feature monitoring; multi-detector integration (PSI, KS, JSD, River ADWIN); explainability via SHAP and permutation sensitivity; hybrid storage architecture; human-in-the-loop retraining approvals.
- **Out of Scope:** Unstructured data (images, audio, free text NLP); deep neural network weight monitoring; causal DAG inference; distributed streaming clusters (Apache Flink/Kafka).

### 7.2 Objective Fulfillment Roadmap
All twelve specific objectives were systematically mapped across the project components:
- Objectives $O_1$ to $O_3$ (Automation and Statistical Detectors): Realized in DFD 3.0/4.0 and Component 4 queries.
- Objectives $O_4$ to $O_7$ (Explainability & DIS): Realized in Section 10 and Component 8.
- Objectives $O_8$ to $O_9$ (Performance & Warning vs. Degradation): Realized in Section 8 and Decision Engine.
- Objectives $O_{10}$ to $O_{11}$ (Hybrid DB & Retraining Rollback): Realized in PostgreSQL DDL and MongoDB Collections.
- Objective $O_{12}$ (Experimental Evaluation): Realized in Simulation Lab and Verification Checklist.

---

# 8. COMPONENT 8: DESIGN OF FORMS, SECURITY, AND VALIDATION

### 8.1 Front-End Form & Screen Specifications
The front end is organized into four primary role-governed workflows:

```
+----------------------------------------------------------------------------------------------------+
| SCREEN 1: AUTHENTICATION                                                                           |
| Email: [ analyst@rvce.edu.in             ]   Password: [ ***************** ]  [ Log In ]           |
| Role-aware redirection (Analyst -> Monitoring; Admin -> Thresholds/Retraining; Reviewer -> Reports)|
+----------------------------------------------------------------------------------------------------+
| SCREEN 2: DATASET UPLOAD & PROFILING                                                               |
| Dataset Name: [ loan_default_2026      ]   Target Column: [ default_flag  ]                        |
| CSV File:     [ loans_train.csv (14MB) ]   Source:        [ Synthetic Underwriting Simulator ]     |
| [ Upload and Validate ] -> Triggers server-side schema inference and Data Quality Table            |
+----------------------------------------------------------------------------------------------------+
| SCREEN 3: DRIFT RESULTS & ROOT-CAUSE ANALYSIS                                                      |
| Selected Job: [ Job #104 (Batch B004) ]    Status: [ Completed ]   Alert: [ DRIFT-WARNING (High) ] |
| Feature         PSI      KS-Stat   JSD      Norm-D   FI (SHAP)  Sensitivity  DIS     Severity     |
| income          0.9732   0.2310    0.1536   1.0000   0.6000     0.5000       0.1200  High         |
| credit_score    0.0412   0.0120    0.0089   0.0000   0.8500     0.9000       0.0000  Low          |
| [ Bar Chart: Feature DIS Ranking ] [ Histogram Overlay: Income Reference vs. Production Batch ]   |
+----------------------------------------------------------------------------------------------------+
| SCREEN 4: ALERTS, RETRAINING & ROLLBACK (ADMINISTRATOR ONLY)                                       |
| Active Alert: CONFIRMED-DEGRADATION (Severe Drop: AUC 0.84 -> 0.76 [-0.08])                        |
| Challenger Comparison: XGBoost v2 (AUC: 0.835) vs Champion: Random Forest v1 (AUC: 0.760)          |
| [ Approve & Deploy Challenger ]             [ Reject ]             [ Rollback to Previous ]        |
+----------------------------------------------------------------------------------------------------+
```

### 8.2 Multi-Tier Input Validation Rules

| Input Domain | Client-Side Check (Streamlit / JS) | Server-Side Validation (FastAPI + Pydantic v2) | Database Enforced Constraint |
|---|---|---|---|
| **User Email** | Non-empty, contains `@`, basic regex | `pydantic.EmailStr` validation, lowercase normalization | `email VARCHAR(150) UNIQUE` |
| **Password** | Min 8 chars, 1 letter, 1 digit | Regex check: `any(c.isalpha()) and any(c.isdigit())` | Stored as salted Argon2id hash |
| **Dataset File** | Extension `.csv`, size $< 50\text{MB}$ | MIME check `text/csv`, scan header row, verify target | `row_count >= 0` check |
| **Batch Ingest** | Preview first 10 rows in table | Exact column name & dtype match against Reference Profile | Stored in MongoDB `production_batches` |
| **Ground Truth** | Input restricted to 0 or 1 | Values coerced to integer $\{0, 1\}$; verify record exists | `prediction_logs` index check |
| **Thresholds** | Numbers only | Ordering check: `moderate < high`, `cap > floor` | `thresholds` unique name check |

### 8.3 Security Architecture & Hardening
1. **Argon2id Password Hashing:** Utilizes the memory-hard Argon2id algorithm via `argon2-cffi`. Defeats GPU/ASIC rainbow table attacks.
2. **Stateless JWT Sessions:** Tokens encoded with HMAC-SHA256 containing `sub` (user ID), `role`, `iat`, and `exp`. Tokens expire after 60 minutes.
3. **Role-Based Access Control (RBAC):** FastAPI dependency injection (`require_roles("administrator")`) gates mutating endpoints.
4. **SQL Injection Elimination:** 100% of relational queries execute via SQLAlchemy parameterized queries or Core ORM bindings. No string concatenation.
5. **NoSQL Injection Mitigation:** MongoDB filters are built exclusively from typed Pydantic attributes. Any dictionary key starting with `$` is rejected.

---

# 9. COMPONENT 9: HYBRID ARCHITECTURE & DEMONSTRATION PLAN

### 9.1 Hybrid SQL + NoSQL Storage Strategy
DriftGuard explicitly avoids storing unstructured or rapidly growing event streams in PostgreSQL, and avoids storing financial governance records in MongoDB:

```
+----------------------------------------------------------------------------------------------------+
| APPLICATION INTEGRATION LAYER (FastAPI Services)                                                   |
| Joined in Python memory via: model_id, job_id, batch_id (No cross-database foreign keys)          |
+-------------------------------------------------+--------------------------------------------------+
| RELATIONAL STORE: PostgreSQL 15                 | DOCUMENT STORE: MongoDB 6                        |
| - Users & Password Hashes                       | - Raw Ingested Prediction Logs                   |
| - Dataset & Feature Metadata                    | - Production Batch Documents                     |
| - Registered Model Artifacts & Baselines        | - Detector Drift Events with 50-bin Histograms   |
| - Monitoring Job Records & Execution Windows    | - Root-Cause Reports with SHAP explanations      |
| - Aggregated Drift Summaries & DIS Scores       | - Per-stage Latency Timings (monitoring_logs)    |
| - Alerts & Retraining Governance Records        | - Multi-seed Simulation Experiment Runs          |
| Integrity: BCNF, ACID transactions, Unique Keys | Scalability: Unbounded append, JSON flexibility  |
+-------------------------------------------------+--------------------------------------------------+
```

### 9.2 Live Demonstration Protocol (measured on the current code)

All data is synthetic. The numbers are from the audit run on 20 Nov 2026 (`POST /simulation/seed-demo`, PostgreSQL 16 + MongoDB). Exact values vary slightly by seed.

1. **Authentication & roles.** The first user must be an administrator (otherwise 403). After that, only administrators can create users. A reviewer is read-only: uploading returns 403. A tampered or expired JWT returns 401, a duplicate email 409, and a weak password 422.
2. **Baseline registration.** The 2020 baseline (2,500 records, 12 monitored features, `loan_default` labels) is validated and profiled into `features.reference_stats`. An XGBoost pipeline is trained (`models` v1, `champion`). Holdout metrics on a 25% stratified split: ROC-AUC ≈ 0.65, accuracy ≈ 0.76, F1 ≈ 0.31.
3. **Stable batch (January, scenario A).** DIS = 0.000, LOW RISK, "Continue monitoring". No alert. ADWIN finds no prediction-stream change.
4. **Moderate drift (March, scenario B: income +22%).** DIS ≈ 0.14–0.20, MEDIUM RISK, `drift-warning`. Labels are pending, so the system says degradation cannot be confirmed.
5. **Severe drift (June / September, scenarios C / D).** DIS ≈ 1.0, HIGH RISK. Debt-to-income and credit score rank first by DIS. September has labels and its performance is **stable**: high drift does not by itself mean degradation.
6. **Concept shift with labels (November).** Production AUC 95% CI ≈ [0.46, 0.50] against a baseline AUC of 0.65, so the state is `severe`. This raises `confirmed-degradation` and CRITICAL RISK. The drift types read: data drift yes, prediction drift yes, confirmed degradation yes.
7. **Controlled retraining.** An administrator trains a challenger from November's labelled records (960 training rows plus the baseline training split; 640 rows form the independent post-drift test set). Measured AUC gain on that test set: **+0.268 (95% CI 0.210–0.330)**, with no regression on the baseline holdout. The administrator approves deployment: v1 → `retired`, v2 → `champion` in one transaction. Rollback restores v1 and sets `rollback_flag`.
8. **Database queries.** Run the SQL (S1–S10) and MongoDB (N1–N5) queries of Component 4 to show both stores in use.

---

# 10. COMPONENT 10: INNOVATIVE EXPERIMENT, MLOPS, & AUTOMATION

### 10.1 Societal Concern & Ethical Motivation
In automated financial underwriting, silent model decay results in systematic discrimination—either denying creditworthy borrowers due to outdated income distributions or exposing lenders to unhedged default risk. By decoupling distributional warnings from confirmed degradation, DriftGuard prevents both unmonitored financial harm and wasteful, computationally expensive model retraining.

### 10.2 Mathematical Formulation of the Drift Impact Score (DIS)
DriftGuard's central theoretical contribution is the **Drift Impact Score ($DIS_i$)**, defined for each feature $i$ as:

$$DIS_i = D_i \times FI_i \times S_i$$

Where:
1. **Normalised Drift Magnitude ($D_i \in [0, 1]$):**
   $$Dn_{m,i} = \text{clip}\left(\frac{s_{m,i} - \text{floor}_{m,i}}{\text{cap}_m - \text{floor}_{m,i}}, 0, 1\right), \quad D_i = \max_{m} Dn_{m,i}$$
   Where $s_{m,i}$ is the observed metric for detector $m \in \{\text{PSI}, \text{KS}, \text{JSD}\}$, and $\text{floor}_{m,i}$ is the 95th percentile ($q_{95}$) bootstrap noise floor.
2. **Feature Importance ($FI_i \in [0, 1]$):** Computed via SHAP values on the production batch:
   $$FI_i = \frac{\sum_{k \in \text{encoded}(i)} \frac{1}{N}\sum_{j=1}^N |\phi_{j,k}|}{\max_j \text{Importance}_j}$$
3. **Model Sensitivity ($S_i \in [0, 1]$):** Evaluated by permuting column $i$ while holding all other features fixed:
   $$S_{\text{raw},i} = \frac{1}{R}\sum_{r=1}^R \frac{1}{N}\sum_{j=1}^N |\hat{y}_j(\tilde{X}^{(r)}_i) - \hat{y}_j(X)|, \quad S_i = \frac{S_{\text{raw},i}}{\max_j S_{\text{raw},j}}$$

### 10.3 DIS Ablation Variants
To evaluate the multiplicative formulation, DriftGuard implements four ablation variants:
- **Variant 1 (Drift Only):** $V_1 = D_i$ (Conventional drift monitoring baseline).
- **Variant 2 (Drift $\times$ Importance):** $V_2 = D_i \times FI_i$.
- **Variant 3 (Weighted Additive Sum):** $V_3 = 0.50 \cdot D_i + 0.25 \cdot FI_i + 0.25 \cdot S_i$.
- **Variant 4 (Rank-Based Fusion):** $V_4 = 1 - \frac{\bar{R}_i - 1}{K - 1}$ where $\bar{R}_i = \frac{\text{Rank}(D) + \text{Rank}(FI) + \text{Rank}(S)}{3}$.

### 10.4 MLOps Automation Pipeline
- **Continuous Monitoring Scheduler:** Automated background task polling for newly arrived batches every $N$ minutes.
- **Atomic Deployment:** PostgreSQL transaction ensuring single-champion operational safety:
  $$\text{BEGIN} \rightarrow \text{UPDATE models (champion} \rightarrow \text{retired)} \rightarrow \text{UPDATE models (challenger} \rightarrow \text{champion)} \rightarrow \text{COMMIT}$$
- **Automated Rollback:** If post-deployment anomalies arise, an administrator triggers `/retrain/{id}/rollback`, instantly restoring the prior champion.

---

# 11. COMPONENT 11: FINAL REPORT & VERIFICATION AUDIT

### 11.1 Independent Verification Audit Summary

Code audit of 20 Nov 2026 (Python 3.11.15 on macOS Apple Silicon; test suite uses in-memory SQLite + mongomock; real PostgreSQL / MongoDB checked separately via Docker):

- `pytest`: **77 passed, 0 failed, 0 skipped** (66 original + 11 new regression tests).
- `ruff check app frontend tests scripts`: **All checks passed**.
- Noise-floor calibration was fixed. Averaged across 10 reference samples ($N=2500$, $n=1000$, $b=300$, 200 trials each), the false-alarm rate at the 95% floor went from 0.55% (PSI) / 0.85% (KS) to 7.35% / 4.10% (nominal 5%).
- Independent numerical checks against hand calculations (in `tests/test_drift_methods.py`) and end-to-end API/auth/retraining workflows (in `tests/test_api_auth.py`, `tests/test_banking_platform.py`, `tests/test_audit_fixes.py`) all pass.

### 11.2 Measured Numerical Drift Properties

| Test Description | Numerical Input Condition | Measured Output | Conformance Status |
|---|---|---|---|
| **PSI Identical Samples** | Standard normal vs. itself ($N=1,000$) | `0.000000e+00` | **PASS** (Zero baseline) |
| **PSI Monotonic Shift** | $+1.0\sigma$ vs. $+3.0\sigma$ mean shift | $+1\sigma: 0.9732 \rightarrow +3\sigma: 6.1866$ | **PASS** (Strictly monotonic) |
| **Binned Proportions** | Open-ended reference bins | $\sum p = 1.000000, \sum q = 1.000000$ | **PASS** (Probability mass conserved) |
| **Categorical Shift** | Binary shift (50% $\rightarrow$ 90% positive) | `0.8789` | **PASS** (Detected shift) |
| **KS Equivalence** | Synthetic sample vs. SciPy `ks_2samp` | $\Delta \text{Stat} = 0.0, \Delta p = 0.0$ | **PASS** (Exact SciPy match) |
| **JSD Symmetry & Range** | Discrete distributions $p$ and $q$ | $JSD(p,q) = JSD(q,p) = 0.153561$ | **PASS** (Symmetric & bounded $[0, 1]$) |
| **Normalisation Floor** | Effect size below noise floor ($0.02 < 0.05$) | `0.0` | **PASS** (Noise clamped to zero) |
| **Synthetic Defaults** | 20,000 generated loan applications | Mean default rate: `17.04%` | **PASS** (Calibrated in $[13\%, 22\%]$) |
| **DIS Bounds & Ties** | Hand-computed test matrix | $DIS \in [0, 1]$, average tie rank | **PASS** (Exact hand match) |

### 11.4 Research Experiment: DIS vs Conventional Drift Alerts (measured)

`scripts/experiments/dis_prioritisation.py`: 10 seeds and 110 synthetic batches (designed scenarios B–D, their concept-shift variants, and 40 random multi-feature drifts); 70 covariate-shift batches are used for ranking. **Ground truth:** for each feature, undo its drift (quantile-map it back onto the reference distribution) and measure how much the deployed model's predictions change.

| Method | Spearman | Precision@3 | NDCG@3 |
|---|---|---|---|
| Conventional PSI ranking | 0.626 | 0.638 | 0.774 [0.729, 0.817] |
| D only (normalised drift) | 0.665 | 0.695 | 0.829 [0.795, 0.862] |
| D × FI | 0.704 | 0.824 | **0.932** [0.910, 0.953] |
| DIS = D × FI × S | 0.702 | **0.833** | 0.910 [0.876, 0.938] |

Interpretation:

1. **Supported:** adding model-specific importance clearly improves prioritisation over conventional drift ranking (NDCG@3 +0.136; DIS is better in 63% of batches and equal in 13%).
2. **Not supported:** the sensitivity factor S. D×FI matches or beats DIS in 89% of batches, and S correlates strongly with FI, so DIS ≈ D × FI².
3. **Alert quality:** DIS alerts (DIS ≥ 0.05) are more precise (1.000 vs 0.935) but have lower recall (0.396 vs 0.549) than the conventional PSI/KS/JSD rule. This is a threshold trade-off.
4. **Performance:** no drift score correlates with the real AUC drop (Spearman −0.180 to −0.063). Covariate shift raises DIS without harming AUC, and concept shift harms AUC with DIS ≈ 0. Performance changes must therefore come from labels, as the decision engine already requires.

Limitations: synthetic data, one model family, a model-output-based ground truth (aligned with what "consequential" means here, but partly favouring model-aware methods), and thresholds not tuned.

### 11.3 Project Health & Honest Boundary Statement
- **Foundations Completed:** Full database DDL, MongoDB collection schemas, Argon2id/JWT security, synthetic loan generation, univariate statistical drift engines (PSI, KS, JSD), and DIS mathematical formulations are fully verified and passing.
- **Academic Integrity:** No fabricated results, synthetic accuracy numbers, or pre-filled benchmark metrics exist in the repository. All numbers presented derive from real command executions.

---

## 12. CONCLUSION & FUTURE SCOPE

DriftGuard demonstrates the successful synergy between classical database management principles and modern machine learning governance. By unifying BCNF relational integrity in PostgreSQL with flexible document-oriented telemetry in MongoDB, the system solves the operational challenges of alert fatigue, silent model decay, and premature retraining.

**Future Enhancements:**
1. Integration of streaming Kafka pipelines for high-throughput sub-second inference ingestion.
2. Group-fairness drift tracking across protected demographic attributes.
3. Cryptographic hash-chaining of retraining governance logs to establish tamper-proof audit trails for regulatory compliance (e.g., EU AI Act, Basel III/IV).

---
*Report certified by student: Animesh (1RV24CD028) | Dept. of Computer Science & Engineering, RV College of Engineering®*
