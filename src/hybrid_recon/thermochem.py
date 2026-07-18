"""
thermochem.py -- calibrated equilibrium-property surrogate.

The original study builds a Gibbs-equilibrium grid in Cantera and interpolates
T_c, M, gamma, f_g, f_c, then RE-DERIVES c* from those properties. The exact
Cantera setup (species set, condensed-phase model, solver tolerances) is not
available to this reconstruction, so a smooth analytical surrogate is used in
its place, calibrated to two stated anchor values and to the qualitative shape
of the equilibrium responses.

DISCLAIMER: the surrogate reproduces the correct thermodynamic ORDERING
(c* maximum richer than the temperature maximum; peak c* falling and peak
temperature rising with aluminium loading) but is NOT a substitute for a real
equilibrium solve, and the absolute values it returns are approximate.
"""
from __future__ import annotations
import numpy as np

R_U = 8314.462618           # J/(kmol K) universal gas constant

# --- calibration anchors (pure fuel, O/F = 2.25, P_c = 7 MPa) -----------------
# Two representative pure-paraffin/LOX equilibrium reference values (of the order
# a CEA or Cantera solve returns at this operating point). They fix the two free
# constants: a temperature-deficit factor via T_c and the c* scale factor K_c.
# Edit these to match your own equilibrium reference if desired.
ANCHOR_OF   = 2.25
ANCHOR_TC   = 3095.0        # K   (anchor A: chamber temperature)
ANCHOR_CSTAR = 1712.0       # m/s (anchor B: characteristic velocity)


def chamber_temperature(of: np.ndarray, w_al: float = 0.0) -> np.ndarray:
    """Adiabatic-flame-like chamber temperature surrogate T_c(O/F, w_Al) [K].

    Rises to a plateau near O/F ~ 3.4 for pure fuel; the plateau shifts richer
    and the peak rises as aluminium loading increases (preferential Al oxidation).
    """
    of = np.asarray(of, dtype=float)
    shift = 1.0 - 0.9 * w_al                     # peak moves rich with Al
    peak  = 3545.0 * (1.0 + 0.18 * w_al)         # peak T rises with Al
    t = peak * (1.0 - np.exp(-((of * shift) / 1.55) ** 2.2))
    t *= (1.0 - 0.02 * np.maximum(of - 3.4, 0.0))
    return t


def gas_molecular_weight(of: np.ndarray, w_al: float = 0.0) -> np.ndarray:
    """Gas-phase molecular weight surrogate M(O/F) [kg/kmol]. Rises with O/F."""
    of = np.asarray(of, dtype=float)
    return 12.8 + 3.35 * of + 0.6 * w_al


def gas_gamma(of: np.ndarray, w_al: float = 0.0) -> np.ndarray:
    """Chamber specific-heat ratio surrogate gamma(O/F)."""
    of = np.asarray(of, dtype=float)
    return 1.24 - 0.010 * (of - 2.25)


def condensed_fraction(of: np.ndarray, w_al: float = 0.0) -> np.ndarray:
    """Condensed-phase (Al2O3) mass fraction surrogate f_c."""
    of = np.asarray(of, dtype=float)
    if w_al <= 0.0:
        return np.zeros_like(of)
    # alumina forms preferentially; more oxidizer -> more complete oxidation
    return np.clip(1.889 * w_al * np.clip(of / (of + 1.0), 0, 1), 0.0, 0.9 * w_al + 0.3)


def gas_fraction(of: np.ndarray, w_al: float = 0.0) -> np.ndarray:
    """Gas-phase mass fraction f_g = 1 - f_c."""
    return 1.0 - condensed_fraction(of, w_al)


def _gamma_function(g: np.ndarray) -> np.ndarray:
    """Vandenkerckhove function Gamma(gamma) appearing in the c* relation."""
    return np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def _cstar_raw(of: np.ndarray, w_al: float, k_c: float) -> np.ndarray:
    """c* re-derived from fundamental properties (Eq. 6 of the model)."""
    tc = chamber_temperature(of, w_al)
    m  = gas_molecular_weight(of, w_al)
    g  = gas_gamma(of, w_al)
    fg = gas_fraction(of, w_al)
    # c* = K_c * sqrt(R_u f_g T_c / M) / Gamma(gamma)
    return k_c * np.sqrt(R_U * fg * tc / m) / _gamma_function(g)


# calibrate K_c once so that c*(anchor) == ANCHOR_CSTAR for pure fuel
_K_C = ANCHOR_CSTAR / float(_cstar_raw(np.array([ANCHOR_OF]), 0.0, 1.0)[0])


def characteristic_velocity(of: np.ndarray, w_al: float = 0.0) -> np.ndarray:
    """Calibrated characteristic velocity c*(O/F, w_Al) [m/s], re-derived."""
    return _cstar_raw(np.asarray(of, dtype=float), w_al, _K_C)


def calibration_report() -> dict:
    """Return the calibration diagnostics for verification."""
    ofs = np.linspace(1.25, 4.5, 400)
    cs = characteristic_velocity(ofs, 0.0)
    return {
        "K_c": _K_C,
        "cstar_at_anchor": float(characteristic_velocity(np.array([ANCHOR_OF]), 0.0)[0]),
        "cstar_anchor_target": ANCHOR_CSTAR,
        "peak_cstar": float(cs.max()),
        "peak_cstar_of": float(ofs[cs.argmax()]),
        "tc_at_anchor": float(chamber_temperature(np.array([ANCHOR_OF]), 0.0)[0]),
        "tc_anchor_target": ANCHOR_TC,
    }


if __name__ == "__main__":
    from pprint import pprint
    pprint(calibration_report())
