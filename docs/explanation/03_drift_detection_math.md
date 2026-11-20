# 03 · Drift Detection Mathematics (With Hand-Verified Code Examples)

Every detector in DriftGuard answers a slightly different question about how a production batch differs from the 2020 baseline. All Python outputs below were produced by running DriftGuard's own functions in `.venv`.

---

## 1. Population Stability Index (PSI)

- **Plain English & analogy:** Imagine sorting 2020 customers into 10 equal buckets by income (each bucket holds 10% of customers). Now drop 2026 customers into those *exact same* dollar brackets. If each bracket still holds ~10%, nothing changed (PSI ≈ 0). If the top bracket now holds 35% and the bottom holds 2%, PSI jumps.
- **Formula:** For bins $k = 1, \dots, K$, let $q_k$ be the reference (baseline) share and $p_k$ be the production share (each clipped to $\ge 10^{-4}$ to prevent $\log(0)$):
  $$\text{PSI} = \sum_{k=1}^{K} (p_k - q_k) \ln\!\left(\frac{p_k}{q_k}\right)$$
- **Worked example by hand (categorical, 3 categories A, B, C):**
  - Baseline $q = [0.5, 0.3, 0.2]$ (50 A, 30 B, 20 C); Production $p = [0.2, 0.3, 0.5]$ (20 A, 30 B, 50 C).
  - Term A: $(0.2 - 0.5) \ln(0.2 / 0.5) = (-0.3)(-0.91629) = 0.274887$.
  - Term B: $(0.3 - 0.3) \ln(1) = 0$.
  - Term C: $(0.5 - 0.2) \ln(0.5 / 0.2) = (0.3)(0.91629) = 0.274887$.
  - Unseen-category bucket: $p_{\text{unseen}}=0 \to 10^{-4}, q_{\text{unseen}}=0 \to 10^{-4}$, contributes $0$.
  - **Total by hand:** $0.274887 + 0 + 0.274887 = \mathbf{0.549774}$.
- **Verified with DriftGuard code (`app/services/drift/psi.py → categorical_psi()`):**
  ```python
  import numpy as np
  from app.services.drift.psi import categorical_psi
  ref = np.array(["A"]*50 + ["B"]*30 + ["C"]*20)
  prod = np.array(["A"]*20 + ["B"]*30 + ["C"]*50)
  round(categorical_psi(ref, prod), 6)  # -> 0.549774  (exact match!)
  ```
- **Code implementation details (`app/services/drift/psi.py`):**
  - `_clean()` drops `NaN`/infinite values and requires at least **30** finite observations (`minimum_rows=30`).
  - `binned_proportions()` builds `settings.psi_bins = 10` quantile edges from the reference, replacing the outer edges with `(-inf, +inf)` so production values outside the 2020 min/max are never silently dropped.
  - Uses **natural log** (`np.log`) and clips both $p, q$ to `1e-4`.
  - `categorical_psi()` adds an explicit unseen-category bucket so brand-new categories in production contribute to PSI.
- **Pros / limitations:** Industry standard in banking credit risk; easy to explain. Limitation: depends on binning and ignores within-bin shape changes.
- **Conventional rule of thumb:** `PSI > 0.25` (`settings.conventional_psi = 0.25`).

---

## 2. Kolmogorov–Smirnov (KS) Two-Sample Test

- **Plain English & analogy:** Line up 2020 incomes from smallest to largest and plot the cumulative percentage (ECDF). Do the same for 2026 incomes. The **KS statistic** is the single largest vertical gap between the two curves (0 = identical, 1 = completely separated).
- **Formula:**
  $$D_{\text{KS}} = \sup_{x} \left| F_{\text{ref}}(x) - F_{\text{prod}}(x) \right|$$
- **Worked example by hand:**
  - Reference: 30 values — six each of `{1, 2, 3, 4, 5}`. Production: 30 values — six each of `{3, 4, 5, 6, 7}`.
  - At $x = 2$: $F_{\text{ref}}(2) = 12/30 = 0.40$, while $F_{\text{prod}}(2) = 0.00$. Gap = $|0.40 - 0.00| = \mathbf{0.40}$ (maximum gap).
- **Verified with DriftGuard code (`app/services/drift/ks.py → ks()`):**
  ```python
  import numpy as np
  from app.services.drift.ks import ks
  x = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 6)
  y = np.repeat([3.0, 4.0, 5.0, 6.0, 7.0], 6)
  ks(x, y)  # -> (0.4, 0.015643390496754332)  (exact match!)
  ```
- **Code implementation details:** Wraps `scipy.stats.ks_2samp(_clean(reference), _clean(production))`. Returns `(None, None)` for categorical features (`numerical=False`), because categories have no numeric ordering.
- **Magnitude vs statistical significance (critical viva point!):**
  - `ks_stat` measures **effect size** (how big the gap is, in `[0, 1]`).
  - `ks_p` is the **p-value** (probability of seeing that gap by chance if there is no drift).
  - **Large-sample trap:** With $n = 10,000$ rows, a tiny, meaningless 1% shift can produce `ks_p = 1e-8 < 0.01` ("statistically significant"!), even though `ks_stat = 0.02` is practically tiny. That is why DriftGuard normalises the **statistic** (`ks_stat`) above its bootstrap noise floor rather than relying on `ks_p < 0.01` alone.
- **Conventional rule of thumb:** `ks_p < 0.01` (`settings.conventional_ks_p = 0.01`).

---

## 3. Jensen–Shannon Divergence (JSD)

- **Plain English & analogy:** KL divergence measures how surprised you are when you expect distribution $Q$ but see $P$, but KL is asymmetric and can blow up to $\infty$. **JSD** averages $P$ and $Q$ into a midpoint $M = \frac{1}{2}(P + Q)$ and measures the average KL distance from both to $M$. It is symmetric and always bounded in $[0, 1]$ (with base-2 log).
- **Formula:**
  $$\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M), \quad M = \frac{P + Q}{2}$$
  where $D_{\text{KL}}(P \parallel M) = \sum_k p_k \log_2(p_k / m_k)$.
- **Worked example by hand:**
  - $q = [0.5, 0.3, 0.2]$, $p = [0.2, 0.3, 0.5]$, so midpoint $m = [0.35, 0.30, 0.35]$.
  - $D_{\text{KL}}(p \parallel m) = 0.2 \log_2(0.2/0.35) + 0 + 0.5 \log_2(0.5/0.35) = -0.16147 + 0.25729 = 0.09582$.
  - By symmetry $D_{\text{KL}}(q \parallel m) = 0.09582$, so $\text{JSD} = \mathbf{0.09582}$.
- **Verified with DriftGuard code (`app/services/drift/jsd.py → jsd_from_counts()`):**
  ```python
  import numpy as np
  from app.services.drift.jsd import jsd_from_counts
  round(jsd_from_counts(np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])), 6)
  # -> 0.095816  (exact match!)
  ```
- **Code implementation details:** Adds `1e-8` smoothing, normalises both vectors to sum to 1, calls `scipy.spatial.distance.jensenshannon(..., base=2) ** 2` (SciPy returns the square root, the JS *distance*, so squaring recovers the divergence in `[0, 1]`). Works on both numerical (using PSI's binned proportions) and categorical features.
- **Conventional rule of thumb:** `JSD > 0.10` (`settings.conventional_jsd = 0.1`).

---

## 4. Bootstrap Noise Floor, Normalisation, D (Max) & Severity

- **Plain English & analogy:** Even if 2026 customers come from the *exact same* distribution as 2020, random sampling noise produces a non-zero PSI (e.g., `0.02`) and KS (`0.04`). Before calling something "drift", we ask: *"How big can PSI/KS/JSD get purely by random luck for this batch size?"* That 95th percentile is the **noise floor**.
- **How `app/services/drift/calibration.py` computes it:**
  - `_draw_pair(ref, n, rng)` draws two samples *with replacement* from the reference: one of size $N$ (full reference size) and one of size $n$ (production batch size), repeated `b = 200` times (`settings.null_bootstrap_b`).
  - Takes the `0.95` quantile (`settings.null_quantile`) of the 200 simulated PSI, KS, and JSD values. Batch sizes are rounded to 2 significant figures via `batch_size_bucket()` so floors can be cached.
- **The Audit Bug Fix (great viva story!):**
  - *Before the audit:* `_draw_pair` drew two **batch-sized** ($n$ vs $n$) samples. That inflated sampling variance ($\sim 2/n$ instead of $1/n + 1/N$), making noise floors too high (~0% false alarms instead of 5%) and biasing $D$ downward.
  - *After the fix:* Drawing $N$ vs $n$ matches the real comparison (full reference $N$ vs batch $n$). Averaged across 10 reference samples ($N=2500, n=1000, b=300$), false-alarm rates moved from `0.55%` (PSI) / `0.85%` (KS) to `7.35%` / `4.10%` (around the nominal 5%).
- **Normalisation (`app/services/drift/normalise.py → normalise()`):**
  $$Dn_m = \text{clip}\!\left(\frac{s_m - \text{floor}_m}{\text{cap}_m - \text{floor}_m},\; 0,\; 1\right)$$
  with caps `norm_cap_psi = 0.5`, `norm_cap_ks = 0.5`, `norm_cap_jsd = 0.3`. Anything at or below the noise floor maps cleanly to `0.0`.
- **Worked numerical example (run through DriftGuard):**
  - Reference: 1,000 draws from $\mathcal{N}(50000, 10000^2)$; Production: 1,000 draws from $\mathcal{N}(60000, 10000^2)$ (+1 SD shift).
  - Observed: `PSI = 0.8304` (floor `0.0329` → `norm = 1.0000`), `KS = 0.3940` (`p = 1.14e-69`, floor `0.0580` → `norm = 0.7602`), `JSD = 0.1398` (floor `0.0059` → `norm = 0.4553`).
  - Aggregated $D = \max(1.0000, 0.7602, 0.4553) = \mathbf{1.0000}$ → `severity = "high"`.
- **Why `max` instead of `mean`, and how disagreement is exposed:**
  - PSI, KS, and JSD have different sensitivities (e.g., KS catches a median shift that coarse bins might blur; PSI catches tail shifts). Averaging them would dilute a genuine shift caught strongly by one detector.
  - Using `max` (`settings.d_aggregation = "max"`) is conservative, and to avoid hiding disagreement, `compute_feature_drift()` attaches `detector_agreement`: `agree-drift` (all detectors > floor), `agree-stable` (none > floor), or `disagree` (only a subset > floor).
- **Severity bands (`app/services/monitoring/analysis.py → severity_of()`):**
  - `low` (`Stable`): $D < 0.15$
  - `moderate` (`Moderate Drift`): $0.15 \le D < 0.50$
  - `high` (`Significant Drift`): $D \ge 0.50$

---

## 5. ADWIN (Adaptive Windowing — Streaming Detector)

- **Plain English & analogy:** Imagine watching a stream of daily temperatures. ADWIN keeps a sliding window of recent values and continuously checks if splitting the window into an "older half" and a "newer half" shows a statistically significant difference in their averages. When it does, it drops the old half and flags a change.
- **Crucial scope boundary:** ADWIN works on a **single bounded numeric stream**, NOT feature histograms. DriftGuard never uses ADWIN for feature-distribution drift. Instead, `app/services/drift/adwin.py → adwin_stream()` runs River's `ADWIN(delta=0.002)` on two explicitly defined model streams:
  1. **`prediction` stream:** predicted default probability $p \in [0, 1]$ per record (detects **prediction drift**, needs no labels).
  2. **`error` stream:** absolute error $|y - p| \in [0, 1]$ per labelled record (detects **error-level shift**, needs labels).
- **Reference segment:** Each stream feeds the champion's **baseline holdout** signal (shuffled with fixed seed 42 so it is stationary) followed by the production batch in record order. Only detections after the boundary count as production detections (`first_detection_index`, 0-based inside production); detections inside the reference segment are counted separately (`n_detections_reference`, ideally `0`).
- **Worked example (verified in code):**
  - 500 draws from $\text{Beta}(2, 8)$ (mean `0.1918`) followed by 500 draws from $\text{Beta}(5, 5)$ (mean `0.4985`):
  - `adwin_stream(...)` → `detected=True`, `first_detection_index=43`, `n_detections_production=1`, `n_detections_reference=0`.
- **Live demo output (`November_2026`, job #14):**
  - `prediction` stream: detected at production record `78` (`reference_mean = 0.353 → production_mean = 0.559`).
  - `error` stream: detected at production record `206` (`reference_mean = 0.374 → production_mean = 0.495`).
