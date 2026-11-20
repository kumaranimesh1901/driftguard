# 08 · Research Experiment & Honest Contribution

## 1. Research Question & Hypothesis

**Research Question:** *Can combining multi-detector statistical drift magnitude with model-specific feature importance and observed empirical performance changes improve the prioritisation of potentially consequential drift compared with conventional unweighted drift alerts?*

**Hypothesis:** Ranking drifted features by model-aware scores ($D \times FI$ or $\text{DIS} = D \times FI \times S$) will put features that actually change credit decisions at the top of the list much more accurately than ranking by conventional PSI or KS alone.

---

## 2. Experiment Design (`scripts/experiments/dis_prioritisation.py`)

- **Scale:** 10 random seeds × 11 synthetic batches per seed = **110 batches** (`3` designed covariate-shift scenarios B, C, D; `4` concept-shift variants A–D; `4` random multi-feature drifts shifting 2–5 random attributes). Feature ranking is evaluated on the **70 covariate-shift batches** (`runtime: 69.9 s`).
- **Counterfactual Ground Truth (used *only* for evaluation, never by the system):**
  - For production batch $X_p$ and feature $i$, we construct $X_p^{(i)}$ by **undoing feature $i$'s drift**: numerical values are rank-preserving quantile-mapped back onto the reference marginal; categorical values are resampled from reference proportions.
  - Then $\text{impact}_i = \text{mean}|f(X_p) - f(X_p^{(i)})|$ and $\text{flip}_i = \text{share of approve/decline decisions that flip}$.
  - A feature is labelled *consequential* if $\text{flip}_i \ge 1\%$ or $\text{impact}_i \ge 0.01$.
- **Evaluation Metrics (with tiny intuitive meanings):**
  - **Spearman $\rho$:** Rank correlation between predicted feature order and true impact order (`1.0` = identical ranking).
  - **Precision@3:** Fraction of the top-3 ranked features that truly belong in the ground-truth top-3 (e.g., 2 of 3 right = `0.667`).
  - **NDCG@3:** Like Precision@3, but rewards putting the #1 most impactful feature at rank 1 rather than rank 3 (`1.0` = ideal).
  - **Top-1 Hit:** Fraction of batches where the #1 ranked feature matches the true #1 impactful feature.
  - **Alert Precision & Recall:** Of flagged features, how many were truly consequential (precision), and of all consequential features, how many were flagged (recall).

---

## 3. Measured Results (`dis_prioritisation_results.json`)

Command: `PYTHONPATH=. python scripts/experiments/dis_prioritisation.py`

### Feature Ranking vs Counterfactual Prediction Impact (70 Batches, Mean [95% Bootstrap CI])

| Method | Spearman | Precision@3 | NDCG@3 | Top-1 Hit |
|---|---:|---:|---:|---:|
| `conventional_psi` | 0.626 | 0.638 | 0.774 `[0.729, 0.817]` | 0.429 |
| `ks_statistic` | 0.611 | 0.648 | 0.791 `[0.748, 0.831]` | 0.429 |
| `d_only` | 0.665 | 0.695 | 0.829 `[0.795, 0.862]` | 0.443 |
| **`d_x_fi`** | **0.704** | 0.824 | **0.932** `[0.910, 0.953]` | **0.786** |
| **`dis` ($D \times FI \times S$)** | 0.702 | **0.833** | 0.910 `[0.876, 0.938]` | 0.643 |
| `weighted_sum` | 0.691 | 0.690 | 0.861 `[0.817, 0.901]` | 0.671 |
| `rank_based` | 0.619 | 0.567 | 0.709 `[0.645, 0.772]` | 0.400 |

### Alert Quality & Batch-Level AUC-Drop Correlation

| Alert Rule | Total Alerts | True Positives | Precision | Recall |
|---|---:|---:|---:|---:|
| `conventional_alerts` (PSI>0.25 / KS p<0.01 / JSD>0.1) | 291 | 272 | `0.935` | `0.549` |
| `dis_alerts` (DIS ≥ 0.05) | 196 | 196 | **`1.000`** | `0.396` |

- **Batch-level Spearman correlation with real AUC drop (all 110 batches):** `overall_dis = -0.106`, `conventional_alert_count = -0.063`, `max_psi = -0.180`.

---

## 4. Honest Interpretation, Limitations & Novelty

### What Is Supported vs Not Supported
1. **Supported:** Adding model-specific SHAP importance (**$D \times FI$**) clearly beats conventional PSI ranking (`NDCG@3`: `0.932` vs `0.774`, paired DIS-vs-PSI gain `+0.136`, DIS better in `62.9%` of batches and equal in `12.9%`).
2. **Not supported:** Permutation sensitivity **$S$** adds no measurable gain over $D \times FI$ (paired DIS vs $D \times FI$ mean `-0.023`; $D \times FI$ matched or beat DIS in **88.6%** of batches) because $S$ correlates strongly with $FI$ (Spearman `0.87–0.93`).
3. **Confirmed governance boundary:** Zero positive correlation between drift scores and real AUC drop (`-0.180` to `-0.063`) proves covariate shift does not automatically lower AUC and concept shift can drop AUC without large input drift. **Labels are indispensable.**

### Limitations & Future Work
- All data is **synthetic** (`app/services/synthetic_data.py`), evaluated on **one model family** (XGBoost).
- The counterfactual ground truth measures model-output change, which aligns with "consequential to the model" but naturally favours model-aware rankings.
- Future work: replace global permutation $S$ with **shift-aware sensitivity** (model response specifically to the *observed* shift), add multivariate drift tests (MMD/classifier two-sample test), and wire the reserved `thresholds` / `threshold_history` tables to the UI (**Not implemented**).

### What Is Novel vs Existing (Defensible Viva Statement)
- **Existing techniques we built on:** PSI, KS, JSD, TreeSHAP (Lundberg & Lee, 2017), ADWIN (Bifet & Gavaldà, 2007 / River), XGBoost, and the champion–challenger pattern.
- **Our contribution:** (1) An **integrated, dual-database governance platform** enforcing relational invariants (`one_champion_per_dataset`, CHECK/FK constraints, cross-DB compensation) alongside NoSQL telemetry; (2) **calibrated noise-floor normalisation** combined with model-aware prioritisation ($D \times FI$); (3) **strict separation** of drift warnings from label-confirmed degradation; and (4) an **honest empirical ablation** showing that $D \times FI$ works while permutation $S$ is redundant.
