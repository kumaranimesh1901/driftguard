"""Research experiment: does DIS prioritise consequential drift better than conventional alerts?

Research question (SPEC): can combining statistical drift magnitude with
model-specific feature importance (and sensitivity) improve the prioritisation
of *potentially consequential* drift compared with conventional drift alerts?

Operational ground truth (computed only for evaluation, never by the system):
for production batch X_p and feature i, build X_p^(i) in which feature i's
drift is *undone*: numerical values are quantile-mapped onto the reference
marginal (rank-preserving, so the joint structure with other features is kept);
categorical values are re-drawn from the reference proportions. Then

    impact_i   = mean |f(X_p) - f(X_p^(i))|          (prediction impact)
    flip_i     = share of approve/decline decisions that flip at the threshold

i.e. how much the observed drift in feature i actually changes the deployed
model's outputs. A feature with large drift but no effect on the model is
*not* consequential; that is the case DIS is meant to de-prioritise.

Batch-level (with labels): does overall DIS / conventional alert count track
the real AUC drop? Concept-shift batches are included, where P(y|x) changes.

Run:  PYTHONPATH=. python scripts/experiments/dis_prioritisation.py [--seeds 10]
Outputs JSON + a Markdown summary in docs/experiments/.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from app.services.datasets import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    MONITORED_FEATURES,
    NUMERICAL_FEATURES,
)
from app.services.monitoring import model as cm
from app.services.monitoring.analysis import analyze_banking_drift
from app.services.synthetic_data import BankingDataGenerator

METHODS = {
    "conventional_psi": lambda f: f["psi"] if f["psi"] is not None else 0.0,
    "ks_statistic": lambda f: f["ks_stat"] if f["ks_stat"] is not None else f["jsd"] or 0,
    "d_only": lambda f: f["variants"]["d_only"],
    "d_x_fi": lambda f: f["variants"]["d_fi"],
    "dis": lambda f: f["variants"]["dis"],
    "weighted_sum": lambda f: f["variants"]["weighted_sum"],
    "rank_based": lambda f: f["variants"]["rank_based"],
}
K = 3


def undo_feature_drift(prod: pd.DataFrame, ref: pd.DataFrame, col: str, rng) -> pd.DataFrame:
    """Return a copy of ``prod`` whose ``col`` follows the reference marginal."""
    out = prod.copy()
    if col in NUMERICAL_FEATURES:
        v = pd.to_numeric(prod[col], errors="coerce").to_numpy(float)
        r = np.sort(pd.to_numeric(ref[col], errors="coerce").dropna().to_numpy(float))
        ok = np.isfinite(v)
        ranks = (pd.Series(v[ok]).rank(method="average").to_numpy() - 0.5) / ok.sum()
        mapped = v.copy()
        mapped[ok] = np.quantile(r, ranks)
        out[col] = mapped
    else:
        props = ref[col].dropna().astype(str).value_counts(normalize=True)
        out[col] = rng.choice(props.index.to_numpy(), len(prod), p=props.to_numpy())
    return out


def random_drift_batch(gen: BankingDataGenerator, n: int, rng) -> tuple[pd.DataFrame, list[str]]:
    """Baseline-like batch with 2-5 randomly chosen features shifted randomly."""
    df = gen.generate_baseline_2020(n)
    k = int(rng.integers(2, 6))
    cols = list(rng.choice(MONITORED_FEATURES, k, replace=False))
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            p = rng.dirichlet(np.ones(4) * 2)
            df[c] = rng.choice(["salaried", "self-employed", "contract", "unemployed"], n, p=p)
            continue
        v = df[c].astype(float)
        mode = rng.integers(0, 3)
        if mode == 0:  # location shift in SD units
            df[c] = v + rng.choice([-1, 1]) * rng.uniform(0.2, 1.2) * v.std()
        elif mode == 1:  # multiplicative
            df[c] = v * rng.uniform(0.5, 1.8)
        else:  # spread change around the median
            df[c] = v.median() + (v - v.median()) * rng.uniform(0.4, 2.0)
    df["credit_score"] = df["credit_score"].clip(300, 900)
    df["debt_to_income"] = df["debt_to_income"].clip(0, 5)
    df["interest_rate"] = df["interest_rate"].clip(0, 1)
    for c in ("age", "loan_tenure", "number_of_existing_loans"):
        df[c] = df[c].clip(lower=0)
    return df, cols


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    ideal = np.sort(gains)[::-1][:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = float(np.sum(ideal * disc))
    return float(np.sum(gains[order] * disc) / idcg) if idcg > 0 else float("nan")


def evaluate_batch(report, ref, prod, model, threshold, rng) -> dict:
    feats = [f for f in report["feature_details"] if f["psi"] is not None]
    base_p = cm.predict_proba(model, prod)
    impact, flips = {}, {}
    for f in feats:
        alt_p = cm.predict_proba(model, undo_feature_drift(prod, ref, f["feature"], rng))
        impact[f["feature"]] = float(np.mean(np.abs(base_p - alt_p)))
        flips[f["feature"]] = float(np.mean((base_p >= threshold) != (alt_p >= threshold)))
    names = [f["feature"] for f in feats]
    gt = np.array([impact[n] for n in names])
    gt_top = set(np.argsort(-gt)[:K])
    out = {"ground_truth_impact": impact, "ground_truth_flip_rate": flips, "methods": {}}
    for m, fn in METHODS.items():
        sc = np.array([fn(f) for f in feats], dtype=float)
        rho = spearmanr(sc, gt).statistic if np.ptp(sc) > 0 else float("nan")
        out["methods"][m] = {
            "spearman": float(rho),
            "precision_at_3": len(set(np.argsort(-sc)[:K]) & gt_top) / K,
            "ndcg_at_3": ndcg_at_k(sc, gt, K),
            "top1_hit": int(np.argmax(sc) == np.argmax(gt)),
        }
    # Alert sets: conventional rule vs DIS-based "moderate impact" rule.
    consequential = {n for n in names if flips[n] >= 0.01 or impact[n] >= 0.01}
    conv = {f["feature"] for f in feats if f["conventional_alert"]}
    disalert = {f["feature"] for f in feats if f["dis"] >= 0.05}
    for label, s in (("conventional_alerts", conv), ("dis_alerts", disalert)):
        tp = len(s & consequential)
        out[label] = {
            "n_alerts": len(s), "true_positive": tp,
            "precision": tp / len(s) if s else None,
            "recall": tp / len(consequential) if consequential else None,
        }
    out["n_consequential"] = len(consequential)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--random-per-seed", type=int, default=4)
    ap.add_argument("--n-baseline", type=int, default=2500)
    ap.add_argument("--n-batch", type=int, default=1500)
    ap.add_argument("--out", default="docs/experiments")
    args = ap.parse_args()
    t0 = time.time()
    rows, batch_rows = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(10_000 + seed)
        base = BankingDataGenerator(seed).generate_baseline_2020(args.n_baseline)
        model, metrics = cm.train_baseline_model(base, seed=seed)
        plan = [(s, False) for s in "BCD"] + [(s, True) for s in "ABCD"]
        plan += [("R", False)] * args.random_per_seed
        for j, (sc, concept) in enumerate(plan):
            gen = BankingDataGenerator(seed * 100 + j + 1)
            if sc == "R":
                prod, shifted = random_drift_batch(gen, args.n_batch, rng)
                gen2 = BankingDataGenerator(seed * 100 + j + 1)
                prod[LABEL_COLUMN] = gen2.generate_production_batch(
                    n=args.n_batch, scenario="A", with_labels=True)[LABEL_COLUMN].to_numpy()
            else:
                prod = gen.generate_production_batch(
                    n=args.n_batch, scenario=sc, with_labels=True, concept_shift=concept)
                shifted = None
            rep = analyze_banking_drift(base, prod, f"s{seed}-{sc}", model, metrics,
                                        floor_key=f"exp{seed}")
            y = prod[LABEL_COLUMN].astype(int).to_numpy()
            p = cm.predict_proba(model, prod)
            auc_drop = metrics["roc_auc"] - roc_auc_score(y, p)
            batch_rows.append({
                "seed": seed, "scenario": sc + ("+concept" if concept else ""),
                "overall_dis": rep["overall_dis"],
                "conventional_alert_count": rep["conventional_alert_count"],
                "max_psi": max(f["psi"] or 0 for f in rep["feature_details"]),
                "auc_drop": float(auc_drop),
                "performance_state": rep["performance_state"],
                "risk_category": rep["risk_category"],
                "prediction_drift": rep["drift_types"]["prediction_drift"],
            })
            if concept or sc == "A":
                continue  # feature ranking evaluated on covariate-shift batches only
            ev = evaluate_batch(rep, base, prod, model, metrics["threshold"], rng)
            ev.update({"seed": seed, "scenario": sc, "shifted_features": shifted})
            rows.append(ev)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for m in METHODS:
        vals = {k: np.array([r["methods"][m][k] for r in rows], float)
                for k in ("spearman", "precision_at_3", "ndcg_at_3", "top1_hit")}
        summary[m] = {k: {"mean": float(np.nanmean(v)),
                          "ci95": [float(x) for x in np.nanquantile(
                              [np.nanmean(np.random.default_rng(i).choice(v, v.size))
                               for i in range(2000)], [0.025, 0.975])]}
                      for k, v in vals.items()}
    paired = np.array([r["methods"]["dis"]["ndcg_at_3"] - r["methods"]["conventional_psi"]["ndcg_at_3"] for r in rows])
    paired_dfi = np.array([r["methods"]["dis"]["ndcg_at_3"] - r["methods"]["d_x_fi"]["ndcg_at_3"] for r in rows])
    alerts = {}
    for label in ("conventional_alerts", "dis_alerts"):
        tp = sum(r[label]["true_positive"] for r in rows)
        na = sum(r[label]["n_alerts"] for r in rows)
        nc = sum(r["n_consequential"] for r in rows)
        alerts[label] = {"alerts": na, "true_positive": tp,
                         "precision": tp / na if na else None, "recall": tp / nc if nc else None}
    bdf = pd.DataFrame(batch_rows)
    batch_corr = {c: float(spearmanr(bdf[c], bdf["auc_drop"]).statistic)
                  for c in ("overall_dis", "conventional_alert_count", "max_psi")}
    result = {
        "config": vars(args), "n_ranking_batches": len(rows), "n_batches": len(batch_rows),
        "runtime_s": round(time.time() - t0, 1), "ranking_summary": summary,
        "paired_ndcg_gain_dis_vs_psi": {
            "mean": float(np.nanmean(paired)), "share_dis_better": float(np.mean(paired > 1e-9)),
            "share_equal": float(np.mean(np.abs(paired) <= 1e-9))},
        "paired_ndcg_gain_dis_vs_d_x_fi": {
            "mean": float(np.nanmean(paired_dfi)), "share_dis_better": float(np.mean(paired_dfi > 1e-9)),
            "share_equal": float(np.mean(np.abs(paired_dfi) <= 1e-9))},
        "alert_quality": alerts,
        "batch_level_spearman_vs_auc_drop": batch_corr,
        "batch_level_by_scenario": bdf.groupby("scenario")[
            ["overall_dis", "conventional_alert_count", "auc_drop"]].mean().round(4).to_dict("index"),
        "per_batch": rows, "batches": batch_rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dis_prioritisation_results.json").write_text(json.dumps(result, indent=1, default=float))
    lines = ["# DIS prioritisation experiment (generated by scripts/experiments/dis_prioritisation.py)", "",
             (f"Seeds: {args.seeds}; ranking batches: {len(rows)}; all batches: {len(batch_rows)}; "
              f"runtime {result['runtime_s']} s. Synthetic data only."), "",
             "## Feature ranking vs counterfactual prediction impact (mean [95% bootstrap CI])", "",
             "| Method | Spearman | Precision@3 | NDCG@3 | Top-1 hit |", "|---|---|---|---|---|"]
    for m, v in summary.items():
        cell = lambda k, v=v: f"{v[k]['mean']:.3f} [{v[k]['ci95'][0]:.3f}, {v[k]['ci95'][1]:.3f}]"
        lines.append(f"| {m} | {cell('spearman')} | {cell('precision_at_3')} | {cell('ndcg_at_3')} | {cell('top1_hit')} |")
    lines += ["", "## Alert quality (consequential = flip rate >= 1% or mean |dp| >= 0.01)", "",
              "| Rule | Alerts | True positives | Precision | Recall |", "|---|---|---|---|---|"]
    for k, a in alerts.items():
        lines.append(f"| {k} | {a['alerts']} | {a['true_positive']} | {a['precision']:.3f} | {a['recall']:.3f} |")
    lines += ["", "## Batch level: Spearman correlation with real AUC drop (labels)", ""]
    lines += [f"- {k}: {v:.3f}" for k, v in batch_corr.items()]
    (out / "dis_prioritisation_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
