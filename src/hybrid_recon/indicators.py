"""
indicators.py -- normalisation, front thinning, and quality indicators.

Hypervolume and IGD+ are taken from pymoo. Objectives are normalised by a
reference ideal/nadir; each run front is thinned to at most `cap` extreme-
preserving farthest points; the held-out reference set is pooled from the
supplied fronts and its non-dominated subset is retained.
"""
from __future__ import annotations
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus


def feasible_objectives(F, G):
    """Return objective rows whose constraints are all satisfied (g<=0)."""
    mask = (np.asarray(G) <= 0).all(axis=1)
    return np.asarray(F)[mask]


def non_dominated(F):
    """Return the non-dominated subset of F (minimisation)."""
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return F
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated[i] = False
        keep[dominated] = False
    return F[keep]


def farthest_point_thin(F, cap=80):
    """Extreme-preserving farthest-point selection down to `cap` points."""
    F = np.asarray(F, dtype=float)
    if len(F) <= cap:
        return F
    idx = [int(np.argmin(F[:, k])) for k in range(F.shape[1])]
    idx = list(dict.fromkeys(idx))
    while len(idx) < cap:
        d = np.min(np.linalg.norm(F[:, None, :] - F[idx][None, :, :], axis=2), axis=1)
        d[idx] = -1
        idx.append(int(np.argmax(d)))
    return F[idx]


def build_reference(fronts, cap=None):
    """Pool feasible non-dominated points from many fronts into a reference set."""
    pooled = np.vstack([f for f in fronts if len(f)]) if any(len(f) for f in fronts) else np.empty((0,))
    nd = non_dominated(pooled)
    if cap and len(nd) > cap:
        nd = farthest_point_thin(nd, cap)
    return nd


def normalise(F, ideal, nadir):
    span = np.where((nadir - ideal) == 0, 1.0, nadir - ideal)
    return (np.asarray(F, dtype=float) - ideal) / span


def indicators(front_norm, ref_norm, hv_point=1.10):
    """Return (HV, IGD+) for a normalised front against a normalised reference."""
    if len(front_norm) == 0:
        return 0.0, np.inf
    rp = np.full(front_norm.shape[1], hv_point)
    hv = float(HV(ref_point=rp)(front_norm))
    igd = float(IGDPlus(ref_norm)(front_norm))
    return hv, igd
