# 04 · Credit Model, SHAP Importance, Sensitivity & DIS

## 1. The Baseline Credit-Risk Model (`app/services/monitoring/model.py`)

DriftGuard monitors a binary credit-risk classifier predicting `loan_default` (`0` = repaid, `1` = defaulted).

### Features (12 Monitored Attributes — `app/services/datasets.py`)
- **11 Numerical:** `age`, `monthly_income`, `credit_score`, `credit_limit`, `outstanding_loan_amount`, `fd_balance`, `debt_to_income`, `loan_amount_requested`, `loan_tenure`, `number_of_existing_loans`, `interest_rate`.
- **1 Categorical:** `employment_type` (`salaried`, `self-employed`, `contract`, `unemployed`).

### Preprocessing & Model Pipeline (`_build_pipeline()`)
1. **Numerical branch:** `SimpleImputer(strategy="median")` handles missing numbers without leaking production data.
2. **Categorical branch:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` safely encodes unseen production categories as all-zeros instead of crashing.
3. **Classifier:** `XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)`.
4. **Class imbalance handling (`fit_pipeline()`):** Defaults are the minority class (~17–20% in the 2020 baseline). `scale_pos_weight = (len(y_tr) - pos) / pos` weights minority defaults proportionally so the model does not lazily predict "repaid" for everyone.
5. **Train/Holdout split & Threshold (`split_labelled()`, `select_threshold()`):**
   - Deterministic 75% / 25% stratified split (`test_size=0.25`, `random_state=42`). Requires at least `MIN_TRAIN_LABELS = 100` labelled rows and both classes.
   - Decision threshold is chosen on a 91-point grid `[0.05, 0.95]` to maximise F1 **on the training split only**, so the 25% holdout metrics remain an honest out-of-sample estimate.
   - Saved to `artifacts/banking_model_ds{id}[_v{n}].joblib` alongside `.holdout.npz` (holdout probabilities & labels used as the ADWIN reference segment).

### Real Measured Holdout Metrics (from Live PostgreSQL `models` Table)

| Model | Evaluation Split | Accuracy | Precision | Recall | F1-score | ROC-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|---|---|---|
| **v1 (2020 baseline)** | 25% holdout (625 rows) | `0.760` | — | — | `0.342` | `0.664` | `[[436, 61], [89, 39]]` |
| **v5 (current champion)** | 40% post-drift test (1,360 rows) | `0.658` | `0.661` | `0.878` | `0.755` | `0.700` | `[[180, 366], [99, 715]]` |

**How to explain these metrics to your professor (using v1's real confusion matrix `[[436, 61], [89, 39]]`, total 625 holdout customers):**
- **TN = 436** (repaid loans correctly predicted repaid), **FP = 61** (repaid loans falsely flagged as default), **FN = 89** (missed defaults), **TP = 39** (caught defaults).
- **Accuracy** $= (436 + 39) / 625 = \mathbf{76.0\%}$ — overall share correct (misleading on imbalanced data).
- **Precision** $= 39 / (39 + 61) = \mathbf{39.0\%}$ — of loans flagged risky, how many actually defaulted.
- **Recall** $= 39 / (39 + 89) = \mathbf{30.5\%}$ — of actual defaults, how many the model caught.
- **F1-score** $= 2 \cdot \frac{0.390 \times 0.305}{0.390 + 0.305} = \mathbf{0.342}$ — harmonic mean balancing precision and recall.
- **ROC-AUC = 0.664** — probability that a randomly chosen defaulter gets a higher predicted default score than a randomly chosen repayer (threshold-independent: `0.5` is coin-flip, `1.0` is perfect).

---

## 2. SHAP Feature Importance ($FI$) (`model.py → shap_importance()`)

- **Intuition (team-game analogy):** Imagine 12 players (features) play a game whose score is a customer's predicted default probability. **SHAP** (SHapley Additive exPlanations) computes each player's fair average contribution across all subsets of teammates.
- **Implementation:**
  1. Subsamples up to `settings.shap_sample_size = 2000` rows from the production batch.
  2. Transforms through `pre` (`ColumnTransformer`) and runs `shap.TreeExplainer(clf).shap_values(xt)` (exact polynomial-time algorithm for gradient-boosted trees).
  3. Computes $\text{mean}(|\phi_{j,k}|)$ across rows for each encoded column.
  4. **One-hot aggregation:** Because `employment_type` expands into 4 one-hot columns (`cat__employment_type_salaried`, etc.), the code sums their mean $|\text{SHAP}|$ back into the single original feature `employment_type`.
  5. **Normalisation:** Divides every feature's score by the maximum feature score so $FI_i \in [0, 1]$ and the most important feature always has $FI = 1.0$.
- **Global vs Local:** Each individual $\phi_{j,k}$ is a *local* explanation (why customer $j$ got score 0.72). Averaging $|\phi_{j,k}|$ across the batch produces a *global* importance ranking ($FI_i$).
- **Crucial caveat:** **SHAP measures model attribution, not real-world causation.** It tells you what the XGBoost model relies on—even if the model learned a spurious proxy.

---

## 3. Permutation Sensitivity ($S$) (`model.py → permutation_sensitivity()`)

- **Plain English:** Take the production batch, compute baseline predictions $p_0$, then randomly shuffle column $i$ across customers (breaking its link to each customer while keeping its marginal distribution intact) and re-score $p_{\text{shuf}}$. Measure $\text{mean}(|p_{\text{shuf}} - p_0|)$. Repeat `settings.sensitivity_repeats = 5` times, average, and divide by the maximum so $S_i \in [0, 1]$.
- **Small worked example:** If shuffling `debt_to_income` changes average predicted default probability by `0.120` (highest of all 12 features) and shuffling `age` changes it by `0.036`, then $S_{\text{dti}} = 0.120 / 0.120 = 1.000$ and $S_{\text{age}} = 0.036 / 0.120 = 0.300$.
- **Honest audit finding (redundancy with $FI$):** On our 5 live demo batches, the Spearman rank correlation between $FI$ and $S$ is **`0.867` to `0.930`** (`January: 0.923`, `March: 0.923`, `June: 0.867`, `September: 0.930`, `November: 0.895`). Because both measure how much the model responds to a feature globally, $S$ is largely redundant with $FI$ (making $\text{DIS} \approx D \times FI^2$).

---

## 4. Drift Impact Score ($\text{DIS} = D \times FI \times S$)

### Worked Example with 3 Real Features from `November_2026` (Job #14, Live DB)

| Feature | PSI | KS | JSD | $D$ | $FI$ | $S$ | $\text{DIS} = D \times FI \times S$ | $D \times FI$ | Weighted Sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `debt_to_income` | 1.939 | 0.345 | 0.194 | **1.000** | **1.000** | **1.000** | **1.000** | 1.000 | 1.000 |
| `outstanding_loan_amount` | 1.035 | 0.413 | 0.171 | **1.000** | **0.888** | **0.994** | **0.883** | 0.888 | 0.971 |
| `employment_type` | 0.358 | n/a | 0.063 | **0.711** | **0.753** | **0.831** | **0.445** | 0.535 | 0.752 |

Check the arithmetic on `employment_type`: $0.711 \times 0.753 \times 0.831 = 0.4449 \approx \mathbf{0.445}$.

### Ablation Variants (`app/services/explain/dis.py → dis_variants()`)
To test whether multiplying all three factors helps, the code computes 5 scores per feature:
1. `dis` $= D \cdot FI \cdot S$ (zero if *any* factor is zero)
2. `d_only` $= D$ (model-blind baseline)
3. `d_fi` $= D \cdot FI$ (drift × SHAP importance, without $S$)
4. `weighted_sum` $= 0.50 D + 0.25 FI + 0.25 S$ (additive; can stay high even when $D = 0$!)
5. `rank_based` $= 1 - (\bar{R}_i - 1)/(K - 1)$ (average rank fusion)

### Batch-Level DIS & Why June, September, and November All Saturate at `1.000`
- **Batch-level DIS** (`analysis.py`): $\text{Overall DIS} = \max_i(\text{DIS}_i)$, categorised by `batch_drift_level()` using `settings.batch_dis_moderate = 0.05` and `settings.batch_dis_high = 0.25`.
- **Why saturation happens:** Because `debt_to_income` is the #1 most important *and* #1 most sensitive feature (`FI = 1.0`, `S = 1.0`), as soon as its PSI crosses the cap `0.50` (June PSI = `1.119`, September = `1.928`, November = `1.939`), its $D$ clips to `1.000`, so $\text{DIS}_{\text{dti}} = 1 \times 1 \times 1 = 1.000$, and `max` saturates at `1.000`. That is why the dashboard also shows the **number of drifting features** (`June: 5/12`, `September: 7/12`, `November: 7/12`) and per-feature PSI/KS/JSD so operators can distinguish severity beyond the cap.
