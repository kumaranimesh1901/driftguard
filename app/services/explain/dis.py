"""Proposed Drift Impact Score and ablations."""

from scipy.stats import rankdata


def dis_variants(
    d: dict[str, float], fi: dict[str, float], s: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Calculate proposed DIS and tie-stable ablation prioritisation scores."""
    names = list(d)
    if set(d) != set(fi) or set(d) != set(s):
        raise ValueError("D, FI, and sensitivity must have identical feature keys")
    if any(not 0 <= value <= 1 for group in (d, fi, s) for value in group.values()):
        raise ValueError("D, FI, and sensitivity values must be in [0, 1]")
    n = len(names)
    ranks = [
        rankdata([-group[name] for name in names], method="average")
        for group in (d, fi, s)
    ]
    result = {}
    for index, name in enumerate(names):
        rank_mean = sum(rank[index] for rank in ranks) / 3
        rank_score = 1.0 if n == 1 else 1 - (rank_mean - 1) / (n - 1)
        result[name] = {
            "dis": d[name] * fi[name] * s[name],
            "d_only": d[name],
            "d_fi": d[name] * fi[name],
            "weighted_sum": 0.5 * d[name] + 0.25 * fi[name] + 0.25 * s[name],
            "rank_based": float(rank_score),
        }
    return result
