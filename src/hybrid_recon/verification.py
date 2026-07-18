"""
verification.py -- numerical checks on the transient model.

  * RK4 time-step convergence order for the final port diameter;
  * both pressure-closure quantities (successive-iterate change and equation
    residual) at the converged state;
  * burn-average quadrature comparison (left-rectangle vs trapezoidal vs Simpson).
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import simpson
from . import model as M


def rk4_order(design, steps=(2.0, 1.0, 0.5, 0.25, 0.125)):
    """Observed convergence order of the final port diameter across time steps."""
    finals = []
    for dt in steps:
        bd = M.integrate_burn(design, dt=dt)
        finals.append(bd["d_p_final"])
    finals = np.array(finals)
    ref = finals[-1]
    rows = []
    for i in range(len(steps) - 2):
        e1 = abs(finals[i] - ref); e2 = abs(finals[i + 1] - ref)
        if e2 > 0:
            order = np.log(e1 / e2) / np.log(steps[i] / steps[i + 1])
            rows.append((steps[i], float(order), float(e1 / abs(ref) * 100)))
    return rows


def residuals(design):
    """Both pressure-closure quantities at the initial state."""
    st = M.solve_pressure(design["m_ox"], design["d_p0"], design["l_g"],
                          design["d_t"], design["w_al"], design["d_al"])
    return dict(iterations=st["iters"], iter_change=st["iter_change"],
                eqn_residual=st["eqn_residual"])


def quadrature_compare(design):
    """Mean thrust and Isp under three quadrature rules."""
    bd = M.integrate_burn(design)
    t, F, mdot = bd["t"], bd["F"], bd["m_dot"]
    G0 = M.G0
    rect = float(np.mean(F[:-1]))
    trap = float(np.trapezoid(F, t) / (t[-1] - t[0]))
    simp = float(simpson(F, x=t) / (t[-1] - t[0]))
    isp_int = float(np.trapezoid(F, t) / (G0 * np.trapezoid(mdot, t)))
    return dict(F_rect=rect, F_trap=trap, F_simpson=simp,
                rect_vs_trap_pct=abs(rect - trap) / trap * 100, Isp_integrated=isp_int)
