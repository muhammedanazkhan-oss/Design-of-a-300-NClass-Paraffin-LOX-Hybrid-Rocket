"""
stats.py -- non-parametric comparison with multiplicity control.

Friedman omnibus test, paired one-sided Wilcoxon signed-rank tests, Holm
adjustment, and matched-pairs rank-biserial effect sizes.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as ss


def friedman(matrix):
    """Friedman test across algorithms. `matrix` is (n_seeds, n_algorithms)."""
    cols = [matrix[:, j] for j in range(matrix.shape[1])]
    chi2, p = ss.friedmanchisquare(*cols)
    return float(chi2), float(p)


def wilcoxon_one_sided(a, b, better="greater"):
    """Paired one-sided Wilcoxon: is `a` better than `b`?

    better='greater' tests a>b (e.g. HV); 'less' tests a<b (e.g. IGD+).
    Returns (p_value, rank_biserial_r).
    """
    d = np.asarray(a) - np.asarray(b)
    alt = "greater" if better == "greater" else "less"
    try:
        res = ss.wilcoxon(a, b, alternative=alt, zero_method="wilcox")
        p = float(res.pvalue)
    except ValueError:
        p = 1.0
    nz = d[d != 0]
    if len(nz) == 0:
        return p, 0.0
    ranks = ss.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum(); r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    r = float((r_plus - r_minus) / total) if total else 0.0
    if better == "less":
        r = -r
    return p, r


def holm(pvals):
    """Holm step-down adjustment. Returns adjusted p-values in input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def wilson_interval(k, n, z=1.959964):
    """Wilson 95% confidence interval for a proportion, as percentages."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100
