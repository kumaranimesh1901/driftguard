"""Adversarial numerical audit of drift detection and explainability components."""

import sys
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import ks_2samp

from app.services.drift.psi import psi, categorical_psi, binned_proportions
from app.services.drift.ks import ks
from app.services.drift.jsd import jsd_from_counts
from app.services.drift.normalise import normalise
from app.services.explain.dis import dis_variants
from app.services.decision import decide

def run_audit():
    print("=== 6.1 PSI AUDIT ===")
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    
    # Identical samples
    val_identical = psi(ref, ref)
    print(f"PSI identical samples: {val_identical:.6e} (near 0)")
    
    # +1 and +3 std shifts
    val_shift_1 = psi(ref, ref + 1.0)
    val_shift_3 = psi(ref, ref + 3.0)
    print(f"PSI +1 std shift: {val_shift_1:.4f}")
    print(f"PSI +3 std shift: {val_shift_3:.4f}")
    print(f"Shift ordering check (+3 > +1): {val_shift_3 > val_shift_1}")
    
    # Proportions sum to 1
    p, q, bins = binned_proportions(ref, ref + 1.0)
    print(f"Proportions sum: p.sum()={p.sum():.6f}, q.sum()={q.sum():.6f}")
    
    # Binary shift: 50% vs 90%
    binary_ref = np.array(["0"] * 500 + ["1"] * 500)
    binary_prod = np.array(["0"] * 100 + ["1"] * 900)
    val_binary = categorical_psi(binary_ref, binary_prod)
    print(f"Binary categorical PSI (50% -> 90%): {val_binary:.4f}")
    
    # 3-level shift
    c3_ref = np.array(["A"] * 333 + ["B"] * 333 + ["C"] * 334)
    c3_prod = np.array(["A"] * 700 + ["B"] * 200 + ["C"] * 100)
    val_c3 = categorical_psi(c3_ref, c3_prod)
    print(f"3-level categorical PSI: {val_c3:.4f}")
    
    # Zero-inflated shift
    zi_ref = np.concatenate([np.zeros(800), rng.exponential(1.0, 200)])
    zi_prod = np.concatenate([np.zeros(200), rng.exponential(2.0, 800)])
    val_zi = psi(zi_ref, zi_prod)
    print(f"Zero-inflated feature PSI: {val_zi:.4f}")
    
    # Constant feature shift
    const_ref = np.ones(500)
    const_prod = np.full(500, 2.0)
    val_const = psi(const_ref, const_prod)
    print(f"Constant feature shift PSI: {val_const:.4f}")
    
    # Adversarial: NaNs and empty arrays
    try:
        ref_with_nan = np.concatenate([ref, [np.nan]])
        val_nan = psi(ref_with_nan, ref)
        print(f"Adversarial NaN handling in PSI: {val_nan:.4f} (passed, dropped NaN)")
    except Exception as e:
        print(f"Adversarial NaN handling in PSI: FAILED with {e}")
        
    try:
        psi(np.array([1.0]*10), np.array([2.0]*10))
    except ValueError as e:
        print(f"Adversarial small sample handling (<30 rows): correctly raised ValueError: {e}")

    print("\n=== 6.2 KS AUDIT ===")
    prod_ks = ref + 0.5
    stat, p_val = ks(ref, prod_ks, numerical=True)
    expected_stat, expected_pval = ks_2samp(ref, prod_ks)
    print(f"KS stat: {stat:.6f}, expected: {expected_stat:.6f}, diff: {abs(stat - expected_stat):.2e}")
    print(f"KS pval: {p_val:.6e}, expected: {expected_pval:.6e}, diff: {abs(p_val - expected_pval):.2e}")
    cat_stat, cat_pval = ks(ref, prod_ks, numerical=False)
    print(f"Categorical KS returns: ({cat_stat}, {cat_pval})")
    
    print("\n=== 6.3 JSD AUDIT ===")
    counts_a = np.array([10, 20, 30, 40])
    counts_b = np.array([40, 30, 20, 10])
    jsd_ab = jsd_from_counts(counts_a, counts_b)
    jsd_ba = jsd_from_counts(counts_b, counts_a)
    print(f"JSD(a, b): {jsd_ab:.6f}, JSD(b, a): {jsd_ba:.6f}, symmetric: {jsd_ab == jsd_ba}")
    jsd_identical = jsd_from_counts(counts_a, counts_a)
    print(f"JSD identical: {jsd_identical:.6e} (near 0)")
    
    # Check square of scipy base-2 distance
    p_smooth = (counts_a + 1e-8) / (counts_a + 1e-8).sum()
    q_smooth = (counts_b + 1e-8) / (counts_b + 1e-8).sum()
    scipy_dist_sq = dist.jensenshannon(p_smooth, q_smooth, base=2) ** 2
    print(f"JSD equals SciPy base-2 dist squared: {abs(jsd_ab - scipy_dist_sq) < 1e-12}")

    print("\n=== 6.4 NORMALISATION AUDIT ===")
    floor = 0.05
    cap = 0.50
    print(f"Normalise below noise floor (0.02): {normalise(0.02, floor, cap)}")
    print(f"Normalise at noise floor (0.05): {normalise(0.05, floor, cap)}")
    print(f"Normalise intermediate (0.275): {normalise(0.275, floor, cap):.4f}")
    print(f"Normalise at cap (0.50): {normalise(0.50, floor, cap)}")
    print(f"Normalise above cap (0.80): {normalise(0.80, floor, cap)}")
    monotonic = all(
        normalise(v1, floor, cap) <= normalise(v2, floor, cap)
        for v1, v2 in zip(np.linspace(0, 1, 20)[:-1], np.linspace(0, 1, 20)[1:])
    )
    print(f"Normalise is monotonic: {monotonic}")
    # Aggregation check
    try:
        from app.services.drift.normalise import aggregate_d
        print("aggregate_d function: EXISTS")
    except ImportError:
        print("aggregate_d (max/mean aggregation across detectors): MISSING from app/services/drift/normalise.py")

    print("\n=== 6.5 NULL CALIBRATION (200 trials) ===")
    # Draw 200 no-drift pairs of size 1000 from standard normal
    null_stats = []
    ref_large = rng.normal(0, 1, 5000)
    ref_edges = np.unique(np.quantile(ref_large, np.linspace(0, 1, 11)))
    
    # Establish noise floor at q95 on reference bootstrap
    boot_stats = []
    for _ in range(200):
        s1 = rng.choice(ref_large, size=1000, replace=True)
        s2 = rng.choice(ref_large, size=1000, replace=True)
        boot_stats.append(psi(s1, s2, edges=ref_edges))
    noise_floor_q95 = np.quantile(boot_stats, 0.95)
    print(f"Bootstrap null noise floor (q95): {noise_floor_q95:.4f}")
    
    # 200 evaluation trials with independent samples from same distribution
    eval_flags = []
    for _ in range(200):
        s_eval = rng.normal(0, 1, 1000)
        stat_eval = psi(ref_large, s_eval, edges=ref_edges)
        eval_flags.append(stat_eval > noise_floor_q95)
    false_flag_rate = np.mean(eval_flags)
    print(f"Measured false-flag rate on 200 trials at q95: {false_flag_rate:.3f} (target ~0.05)")

    print("\n=== 6.6 SHAP IMPORTANCE AUDIT ===")
    try:
        from app.services.explain.shap_importance import compute_shap_importance
        print("shap_importance module: EXISTS")
    except ImportError:
        print("shap_importance module: MISSING (app/services/explain/shap_importance.py does not exist)")

    print("\n=== 6.7 SENSITIVITY AUDIT ===")
    try:
        from app.services.explain.sensitivity import compute_sensitivity
        print("sensitivity module: EXISTS")
    except ImportError:
        print("sensitivity module: MISSING (app/services/explain/sensitivity.py does not exist)")

    print("\n=== 6.8 DIS AND VARIANTS AUDIT ===")
    d_dict = {"income": 0.4, "credit_score": 0.8, "age": 0.1}
    fi_dict = {"income": 0.6, "credit_score": 0.5, "age": 0.2}
    s_dict = {"income": 0.5, "credit_score": 0.9, "age": 0.3}
    variants = dis_variants(d_dict, fi_dict, s_dict)
    print("Computed DIS variants for toy inputs:")
    for feat, scores in variants.items():
        print(f"  {feat}: {scores}")
    # Verify bounds
    all_in_01 = all(
        0.0 <= val <= 1.0
        for feat in variants.values()
        for val in feat.values()
    )
    print(f"All DIS variant scores in [0, 1]: {all_in_01}")
    # Hand calculation check for 'income':
    # dis = 0.4 * 0.6 * 0.5 = 0.12
    # d_only = 0.4
    # d_fi = 0.4 * 0.6 = 0.24
    # weighted_sum = 0.5 * 0.4 + 0.25 * 0.6 + 0.25 * 0.5 = 0.2 + 0.15 + 0.125 = 0.475
    inc = variants["income"]
    hand_match = (
        abs(inc["dis"] - 0.12) < 1e-6
        and abs(inc["d_only"] - 0.4) < 1e-6
        and abs(inc["d_fi"] - 0.24) < 1e-6
        and abs(inc["weighted_sum"] - 0.475) < 1e-6
    )
    print(f"Hand-computed example matches exactly: {hand_match}")

    print("\n=== 6.9 ADWIN AUDIT ===")
    try:
        from app.services.drift.adwin_monitor import AdwinDetector
        print("ADWIN detector module: EXISTS")
    except ImportError:
        print("ADWIN detector module: MISSING (app/services/drift/adwin_monitor.py does not exist)")

    print("\n=== 6.10 PERFORMANCE STATE AUDIT ===")
    try:
        from app.services.performance import evaluate_performance
        print("Performance evaluation module: EXISTS")
    except ImportError:
        print("Performance evaluation module: MISSING (app/services/performance.py does not exist)")

    print("\n=== 6.11 DECISION ENGINE AUDIT (Section 13 table) ===")
    drift_levels = ["low", "moderate", "high"]
    perf_states = ["unknown", "stable", "degraded", "severe"]
    print("Testing all 12 combinations of (drift_level, perf_state):")
    for d_lvl in drift_levels:
        for p_st in perf_states:
            alert_type, level, rec = decide(d_lvl, p_st)
            print(f"  ({d_lvl:8s}, {p_st:8s}) -> alert_type: {str(alert_type):22s} | level: {level:8s} | rec: {rec}")
            
    # Check specific wording for stable performance
    _, _, rec_mod_stable = decide("moderate", "stable")
    _, _, rec_high_stable = decide("high", "stable")
    print(f"\nChecking stable-performance wording:")
    print(f"  moderate, stable rec: '{rec_mod_stable}'")
    print(f"    Expected: 'Investigate and monitor closely'")
    print(f"    Contains '; labels needed to confirm impact': {'; labels needed to confirm impact' in rec_mod_stable}")
    print(f"  high, stable rec: '{rec_high_stable}'")
    print(f"    Expected: 'Investigate affected features and data quality'")
    print(f"    Contains '; labels needed to confirm impact': {'; labels needed to confirm impact' in rec_high_stable}")

if __name__ == "__main__":
    run_audit()
