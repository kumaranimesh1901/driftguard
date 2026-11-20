"""Banking risk category and recommended actions.

Turns the decision engine output (``decide``: batch drift level x model
performance state) into a credit-risk category and committee actions:

- LOW      batch drift level low
- MEDIUM   batch drift level moderate
- HIGH     batch drift level high, or confirmed degradation without strong drift
- CRITICAL confirmed degradation (labels) with moderate/high drift, or a severe drop

Drift alone never exceeds HIGH: CRITICAL requires labelled evidence.
"""

from typing import Any

from app.core.config import settings
from app.services.recommendations.decision import decide

RISK_CATEGORIES = ("LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK")


def risk_assessment(
    drift_level: str,
    perf: str,
    feature_results: list[dict[str, Any]],
    performance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map the decision engine output to a banking risk category and actions.

    Uses ``decide`` unchanged. CRITICAL requires *confirmed* degradation
    (labels) together with moderate/high drift, or a severe performance drop.
    """
    alert_type, alert_level, recommendation = decide(drift_level, perf)
    ranked = sorted(feature_results, key=lambda f: f.get("dis", 0.0), reverse=True)
    affected = [f["label"] for f in ranked if f["severity"] != "low"]
    top = ranked[:3]

    if alert_type == "confirmed-degradation" and (
        drift_level != "low" or perf == "severe"
    ):
        category = "CRITICAL RISK"
    elif alert_type == "confirmed-degradation" or drift_level == "high":
        category = "HIGH RISK"
    elif drift_level == "moderate":
        category = "MEDIUM RISK"
    else:
        category = "LOW RISK"

    # Features whose distribution moved (D severity) even though their model
    # impact (DIS) kept the batch level low: not a model risk, but possibly a
    # data-quality or pipeline issue that must not be reported as "stable".
    low_impact_drift = [
        f["label"]
        for f in ranked
        if f["severity"] == "high" and f.get("dis", 0.0) < settings.batch_dis_moderate
    ]
    if category == "LOW RISK":
        actions = (
            [
                "Distributions remain relatively stable against the 2020 baseline.",
                "Continue normal monitoring cadence.",
            ]
            if not affected
            else [
                (
                    "Some attributes changed distribution, but the changes have low "
                    "estimated model impact (DIS below the moderate threshold)."
                ),
                "Continue monitoring; no model action is indicated by DIS.",
            ]
        )
    elif category == "MEDIUM RISK":
        actions = [
            "Some financial attributes show statistically significant changes.",
            "Increase monitoring frequency for subsequent production batches.",
            "Investigate affected features: " + (", ".join(affected[:5]) or "none"),
        ]
    elif category == "HIGH RISK":
        actions = [
            (
                "Multiple important financial attributes show substantial drift."
                if drift_level == "high"
                else "Labelled outcomes show performance degradation without strong input drift."
            ),
            (
                "Review credit risk thresholds and loan sanctioning policy assumptions "
                "(no thresholds are changed automatically)."
            ),
            "Conduct additional model evaluation on recent applications.",
            "Notify the credit risk committee.",
        ]
    else:
        actions = [
            "Significant drift combined with confirmed model performance degradation.",
            "Urgent model review by the model risk / validation team.",
            (
                "Initiate a controlled retraining evaluation (champion–challenger); "
                "DriftGuard never retrains or changes sanctioning thresholds automatically."
            ),
            "Notify the credit risk committee immediately.",
        ]
    if low_impact_drift:
        actions.append(
            "Check data quality / upstream pipelines for strongly shifted but "
            "low-impact attributes: " + ", ".join(low_impact_drift[:5])
        )
        if alert_type is None:
            recommendation = (
                recommendation
                + "; verify data quality of strongly drifted low-impact features"
            )
    if perf == "unknown":
        actions.append(
            f"Collect production default outcomes (at least {settings.min_labels} "
            "labelled records) to confirm or rule out performance degradation."
        )

    evidence = []
    for f in top:
        dn = f.get("normalised") or {}
        detectors = ", ".join(
            f"{k.upper()}={v:.3f}"
            for k, v in (("psi", f["psi"]), ("ks", f["ks_stat"]), ("jsd", f["jsd"]))
            if v is not None
        )
        evidence.append(
            f"{f['label']}: DIS={f.get('dis', 0):.3f} (D={f['d']:.2f}, "
            f"FI={f.get('fi', 0):.2f}, S={f.get('s', 0):.2f}); {detectors}"
            + (
                f"; normalised {', '.join(f'{k}={v:.2f}' for k, v in dn.items())}"
                if dn
                else ""
            )
        )
    if performance and performance.get("auc_ci"):
        lo, hi = performance["auc_ci"]
        evidence.append(
            f"AUC drop vs baseline: {performance['auc_drop']:+.3f} "
            f"(95% CI of production AUC [{lo:.3f}, {hi:.3f}])"
        )
    return {
        "alert_type": alert_type,
        "alert_level": alert_level,
        "recommendation": recommendation,
        "risk_category": category,
        "recommended_actions": actions,
        "evidence": evidence,
        "affected_features": affected,
        "low_impact_drift_features": low_impact_drift,
    }
