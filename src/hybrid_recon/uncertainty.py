"""
uncertainty.py -- parametric Monte-Carlo stress test with per-constraint failures.

Propagates assumed input distributions around a nominal design without
re-optimisation and reports overall feasibility, per-constraint violation
probabilities, and the most common joint-violation patterns. Distributions are
engineering stress-test assumptions, not manufacturing-derived statistics.
"""
from __future__ import annotations
import numpy as np
from . import model as M, objectives, thermochem as tc

CONSTRAINT_NAMES = ["thrust_lo", "thrust_hi", "pc_lo", "pc_hi", "of_lo", "of_hi",
                    "gox_lo", "gox_hi", "web", "case_temp", "exit_ratio"]


def run(nominal: dict, n_samples=2000, seed=0):
    """Return dict with feasibility, per-constraint failure probabilities, joints."""
    rng = np.random.default_rng(seed)
    fails = np.zeros(11, dtype=int)
    joint = {}
    n_feasible = 0
    # keep the base callables to restore after each perturbed evaluation
    base_a = M.regression_coeff
    base_cstar = tc.characteristic_velocity
    for _ in range(n_samples):
        s = rng.standard_normal(11)
        d = dict(nominal)
        d["m_ox"] = nominal["m_ox"] * (1 + 0.02 * s[0])
        d["d_p0"] = max(nominal["d_p0"] + 0.10e-3 * s[1], 5e-3)
        d["l_g"] = nominal["l_g"] + 0.5e-3 * s[2]
        d["d_t"] = nominal["d_t"] + 0.05e-3 * s[3]
        d["d_g"] = nominal["d_g"] + 0.10e-3 * s[4]
        d["t_ins"] = nominal["t_ins"] + 0.10e-3 * s[5]
        k_reg = 1 + 0.08 * s[6]
        k_cs = 1 + 0.02 * s[7]
        M.regression_coeff = lambda w, dd, _b=base_a, _k=k_reg: _b(w, dd) * _k
        tc.characteristic_velocity = lambda of, w=0.0, _c=base_cstar, _k=k_cs: _c(of, w) * _k
        try:
            _, g, _ = objectives.evaluate(d)
        except Exception:
            M.regression_coeff = base_a; tc.characteristic_velocity = base_cstar
            continue
        M.regression_coeff = base_a; tc.characteristic_velocity = base_cstar
        viol = g > 0
        fails += viol.astype(int)
        vset = tuple(sorted(np.array(CONSTRAINT_NAMES)[viol].tolist()))
        if not viol.any():
            n_feasible += 1
            key = "FEASIBLE"
        else:
            key = ",".join(vset)
        joint[key] = joint.get(key, 0) + 1
    return dict(
        n_samples=n_samples,
        feasibility_pct=100.0 * n_feasible / n_samples,
        per_constraint_pct={CONSTRAINT_NAMES[i]: 100.0 * fails[i] / n_samples for i in range(11)},
        joint=dict(sorted(joint.items(), key=lambda kv: -kv[1])),
    )
