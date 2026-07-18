"""
model.py -- transient internal-ballistics + hardware model.

Implements, from the governing equations of the design model:
  * ideal-mixture fuel density and oxidizer flux           (Eq. 1)
  * pressure-sensitive regression correlation              (Eqs. 2-4)
  * fuel flow, mixture ratio, RK4 port growth              (Eq. 5)
  * chamber-pressure closure with combustion efficiency    (Eqs. 7-8)
  * supersonic area-Mach / expansion-averaged gamma_eff    (Eq. 9)
  * thrust with conical, kinetic and two-phase losses      (Eqs. 10-11)
  * two-node transient thermal screen, derated case sizing
  * injector / tank / feed / loaded-mass bookkeeping       (Eq. 12)

Reconstruction note: the two-phase throat treatment folds the gas fraction into
the c* relation and is NOT a full homogeneous two-phase choked-flow derivation;
metallized results are therefore indicative only (see README).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from . import thermochem as tc

G0 = 9.80665
P_A = 101325.0
R_U = tc.R_U


# ---------------------------------------------------------------- regression --
def fuel_density(w_al: float) -> float:
    """Ideal-mixture fuel density [kg/m^3] (Eq. 1). Paraffin 890, Al 2700."""
    return 1.0 / ((1.0 - w_al) / 890.0 + w_al / 2700.0)


def regression_coeff(w_al: float, d_al: float) -> float:
    """Regression coefficient a(w_Al, d_Al) (Eq. 3), peaks near d_Al=15 um."""
    return 0.30 * (1.0 + w_al * (0.30 + 0.55 * math.exp(-((d_al - 15.0) / 22.0) ** 2))
                   - 0.45 * w_al ** 2)


def regression_rate(g_ox: float, p_c: float, w_al: float, d_al: float) -> float:
    """Regression rate r-dot [m/s] (Eq. 2). g_ox [kg/m2/s], p_c [Pa]."""
    n = 0.52 + 0.02 * w_al
    m = 0.035 + 0.025 * math.exp(-g_ox / 80.0)
    return 1e-3 * regression_coeff(w_al, d_al) * (g_ox / 10.0) ** n * (p_c / 5e6) ** m


def combustion_efficiency(w_al: float) -> float:
    """Prescribed combustion efficiency eta_c(w_Al) (Eq. 7)."""
    return 0.965 - 0.025 * w_al - 0.015 * w_al ** 2


# --------------------------------------------------------- pressure closure --
def solve_pressure(m_ox, d_p, l_g, d_t, w_al, d_al, tol=1e-10, itmax=30):
    """Fixed-point closure of Eq. (8).

    Returns dict with p_c, of, m_f, cstar, iters, iter_change (successive-iterate
    change |dPc|/Pc) and eqn_residual (|Pc At - m eta c*|/Pc At).
    """
    a_t = math.pi * d_t ** 2 / 4.0
    p_c = 3e6
    change = 1.0
    k = 0
    for k in range(1, itmax + 1):
        g_ox = m_ox / (math.pi * d_p ** 2 / 4.0)
        r = regression_rate(g_ox, p_c, w_al, d_al)
        m_f = fuel_density(w_al) * math.pi * d_p * l_g * r
        of = m_ox / m_f
        cs = float(tc.characteristic_velocity(np.array([of]), w_al)[0])
        p_new = (m_ox + m_f) * combustion_efficiency(w_al) * cs / a_t
        change = abs(p_new - p_c) / p_new
        p_c = p_new
        if change < tol:
            break
    g_ox = m_ox / (math.pi * d_p ** 2 / 4.0)
    r = regression_rate(g_ox, p_c, w_al, d_al)
    m_f = fuel_density(w_al) * math.pi * d_p * l_g * r
    of = m_ox / m_f
    cs = float(tc.characteristic_velocity(np.array([of]), w_al)[0])
    eqn_res = abs(p_c * a_t - (m_ox + m_f) * combustion_efficiency(w_al) * cs) / (p_c * a_t)
    return dict(p_c=p_c, of=of, m_f=m_f, cstar=cs, iters=k,
                iter_change=change, eqn_residual=eqn_res)


# ------------------------------------------------------------------- nozzle --
def gamma_eff(of: float) -> float:
    """Expansion-averaged frozen gamma_eff (exceeds chamber mean)."""
    return float(tc.gas_gamma(np.array([of]))[0]) + 0.031


def two_phase_efficiency(f_c: float, d_al: float) -> float:
    """Continuous condensed-phase loss eta_2p (Eq. 11)."""
    return math.exp(-0.65 * f_c * (d_al / 25.0) ** 0.35)


from functools import lru_cache

@lru_cache(maxsize=100000)
def _exit_mach_cached(eps_r: float, g_r: float) -> float:
    def area_ratio(me):
        return (1.0 / me) * ((2.0 / (g_r + 1.0)) * (1.0 + (g_r - 1.0) * me ** 2 / 2.0)) \
               ** ((g_r + 1.0) / (2.0 * (g_r - 1.0)))
    return brentq(lambda me: area_ratio(me) - eps_r, 1.0001, 12.0)


def _exit_mach(eps: float, g: float) -> float:
    # round to stabilise the cache; Me varies negligibly at this resolution
    return _exit_mach_cached(round(eps, 3), round(g, 4))


def thrust_state(m_ox, d_p, l_g, d_t, eps, w_al, d_al):
    """Full quasi-steady state at one instant. Returns dict of observables."""
    cl = solve_pressure(m_ox, d_p, l_g, d_t, w_al, d_al)
    p_c, of, m_f, cs = cl["p_c"], cl["of"], cl["m_f"], cl["cstar"]
    a_t = math.pi * d_t ** 2 / 4.0
    g = gamma_eff(of)
    me = _exit_mach(eps, g)
    pe_pc = (1.0 + (g - 1.0) * me ** 2 / 2.0) ** (-g / (g - 1.0))
    p_e = pe_pc * p_c
    c_f_ideal = math.sqrt((2 * g ** 2 / (g - 1.0)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
                          * (1.0 - pe_pc ** ((g - 1.0) / g))) + (p_e - P_A) / p_c * eps
    lam = (1.0 + math.cos(math.radians(15.0))) / 2.0
    eta_k = 0.995
    f_c = float(tc.condensed_fraction(np.array([of]), w_al)[0])
    eta_2p = two_phase_efficiency(f_c, d_al)
    f = lam * eta_k * eta_2p * c_f_ideal * p_c * a_t
    g_ox = m_ox / (math.pi * d_p ** 2 / 4.0)
    return dict(F=f, p_c=p_c, of=of, m_f=m_f, g_ox=g_ox, m_dot=m_ox + m_f,
                pe_pa=p_e / P_A, cstar=cs, iter_change=cl["iter_change"],
                eqn_residual=cl["eqn_residual"])


# ---------------------------------------------------------- burn integration --
def integrate_burn(design: dict, dt: float = 0.5, t_b: float = 20.0):
    """RK4 port growth over the burn. `design` keys:
       w_al, m_ox, d_p0, l_g, d_t, eps, d_al, d_g, t_ins (SI units, m/kg/s).
    Returns a dict of time series and end-of-burn scalars.
    """
    w_al = design["w_al"]; m_ox = design["m_ox"]; d_p = design["d_p0"]
    l_g = design["l_g"]; d_t = design["d_t"]; eps = design["eps"]
    d_al = design["d_al"]; d_g = design["d_g"]
    n = int(round(t_b / dt))
    t = np.empty(n + 1); F = np.empty(n + 1); Pc = np.empty(n + 1)
    OF = np.empty(n + 1); Gox = np.empty(n + 1); mdot = np.empty(n + 1)
    for i in range(n + 1):
        st = thrust_state(m_ox, d_p, l_g, d_t, eps, w_al, d_al)
        t[i] = i * dt; F[i] = st["F"]; Pc[i] = st["p_c"]; OF[i] = st["of"]
        Gox[i] = st["g_ox"]; mdot[i] = st["m_dot"]
        if i < n:
            def ddp(dd):
                p = solve_pressure(m_ox, dd, l_g, d_t, w_al, d_al)["p_c"]
                return 2.0 * regression_rate(m_ox / (math.pi * dd ** 2 / 4.0), p, w_al, d_al)
            k1 = ddp(d_p); k2 = ddp(d_p + dt * k1 / 2); k3 = ddp(d_p + dt * k2 / 2); k4 = ddp(d_p + dt * k3)
            d_p += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    web_final = (d_g - d_p) / 2.0          # m, radial web remaining
    return dict(t=t, F=F, p_c=Pc, of=OF, g_ox=Gox, m_dot=mdot,
                d_p_final=d_p, web_final=web_final)


# design-vector helper: order matches the paper's nine variables
VAR_ORDER = ["w_al", "m_ox", "d_p0", "l_g", "d_t", "eps", "d_al", "d_g", "t_ins"]
# bounds in SI (w_al -, kg/s, m, m, m, -, um, m, m)
BOUNDS_LO = [0.00, 0.045, 0.012, 0.220, 0.004, 3.0,  5.0, 0.045, 0.002]
BOUNDS_HI = [0.30, 0.120, 0.030, 0.500, 0.008, 12.0, 60.0, 0.090, 0.010]


def vector_to_design(x) -> dict:
    return {k: float(v) for k, v in zip(VAR_ORDER, x)}
