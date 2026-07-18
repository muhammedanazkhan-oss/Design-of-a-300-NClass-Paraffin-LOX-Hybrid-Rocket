"""
objectives.py -- five objectives and eleven constraints.

Objectives (minimised form, Eq. 13):
    [ -mean_Isp, loaded_mass, T_case_max/423.15, sigma_F/mean_F, (OF_max-OF_min)/OF_mean ]

Constraints (11 inequalities): mean-thrust band (2), chamber-pressure band (2),
mixture-ratio band (2), oxidiser-flux band (2), final web, peak case temperature,
exit-pressure ratio. Seven are trajectory-wide (checked at every recorded state);
the mean-thrust band, final web and peak case temperature are burn-level scalars.

The thermal, structural and mass sub-models are implemented from the described
lumped correlations. They are screening-level and self-consistent, not validated
hardware models.
"""
from __future__ import annotations
import math
import numpy as np
from . import model as M

G0 = M.G0

# ---- material / structural constants ----------------------------------------
RHO_AL = 2700.0                 # 6061 density kg/m3
RHO_INS = 1100.0                # insulation/liner density kg/m3
RHO_LOX = 1141.0                # liquid oxygen kg/m3
K_INS = 0.27                    # insulation conductivity W/m/K
C_CASE = 896.0                  # 6061 specific heat J/kg/K
C_INS = 1500.0                  # insulation specific heat J/kg/K
J_MAT = 1.10                    # material knockdown
J_PROOF = 1.50                  # proof pressure factor
T_AMB = 298.15


def yield_6061(t_k: float) -> float:
    """Temperature-derated 6061-T6 yield strength [Pa]."""
    if t_k <= 298.0:
        return 276e6
    if t_k >= 423.0:
        # linear decay continues above 423 K
        return max(195e6 - (t_k - 423.0) * 1.5e6, 40e6)
    return (276e6 + (195e6 - 276e6) * (t_k - 298.0) / (423.0 - 298.0))


def transient_thermal(p_c_hist, of_hist, d_p_hist, l_g, t_ins, t_case, dt):
    """Two-node implicit energy balance -> peak case temperature [K].

    Node 1: insulation/liner. Node 2: aluminium case. Gas-side convection scales
    with pressure; radiation with T_c^4; external loss to ambient.
    """
    # representative chamber temperature from O/F history
    from . import thermochem as tc
    n = len(p_c_hist)
    d_mean = float(np.mean(d_p_hist))
    a_in = math.pi * d_mean * l_g                      # inner gas-side area
    m_ins = RHO_INS * math.pi * d_mean * l_g * t_ins
    m_cas = RHO_AL * math.pi * (d_mean + 2 * t_ins) * l_g * t_case
    T1 = T_AMB; T2 = T_AMB; T2max = T_AMB
    for i in range(n):
        p_c = p_c_hist[i]; of = of_hist[i]
        Tc_gas = float(tc.chamber_temperature(np.array([of]))[0])
        h_g = 210.0 * (p_c / 5e6) ** 0.8
        q_conv = h_g * a_in * (Tc_gas - T1)
        q_rad = 30e3 * (Tc_gas / 3000.0) ** 4 * a_in
        q_12 = (K_INS / max(t_ins, 1e-4)) * a_in * (T1 - T2)
        q_ext = 12.0 * a_in * (T2 - T_AMB)
        T1 += dt * (q_conv + q_rad - q_12) / max(m_ins * C_INS, 1e-6)
        T2 += dt * (q_12 - q_ext) / max(m_cas * C_CASE, 1e-6)
        T2max = max(T2max, T2)
    return T2max


def size_case(p_c_max, d_p_mean, t_ins, t_case_temp):
    """Iterate thin-wall hoop sizing with the derated allowable. Returns t_case [m]."""
    r = (d_p_mean + 2 * t_ins) / 2.0
    t = 0.0008
    for _ in range(40):
        sigma_allow = yield_6061(t_case_temp) / J_MAT
        t_new = J_PROOF * p_c_max * r / sigma_allow + 0.15e-3
        t_new = max(t_new, 0.0008)
        if abs(t_new - t) < 1e-7:
            t = t_new; break
        t = t_new
    return t


def evaluate(design: dict, dt: float = 0.5, t_b: float = 20.0):
    """Evaluate a design. Returns (objectives[5], constraints[11], info dict).

    Constraint convention: g_i <= 0 is feasible.
    """
    bd = M.integrate_burn(design, dt=dt, t_b=t_b)
    t, F, Pc, OF, Gox, mdot = bd["t"], bd["F"], bd["p_c"], bd["of"], bd["g_ox"], bd["m_dot"]
    web = bd["web_final"]

    # burn averages (trapezoidal quadrature)
    F_mean = float(np.trapezoid(F, t) / (t[-1] - t[0]))
    Isp = float(np.trapezoid(F, t) / (G0 * np.trapezoid(mdot, t)))
    F_cov = float(np.std(F) / F_mean)                       # thrust dispersion (CoV)
    of_range = float((OF.max() - OF.min()) / OF.mean())     # O/F dispersion

    # hardware sizing
    d_p_mean = float(np.mean([design["d_p0"], bd["d_p_final"]]))
    p_c_max = float(Pc.max())
    # couple thermal and case thickness
    t_case = 0.0018
    T_case_max = T_AMB
    for _ in range(3):
        T_case_max = transient_thermal(Pc, OF, np.linspace(design["d_p0"], bd["d_p_final"], len(Pc)),
                                       design["l_g"], design["t_ins"], t_case, dt)
        t_case = size_case(p_c_max, d_p_mean, design["t_ins"], T_case_max)

    # masses
    rho_f = M.fuel_density(design["w_al"])
    vol_fuel = math.pi / 4 * (design["d_g"] ** 2 - design["d_p0"] ** 2) * design["l_g"]
    m_fuel = rho_f * vol_fuel
    m_lox = design["m_ox"] * t_b
    m_case = RHO_AL * math.pi * (design["d_g"]) * design["l_g"] * t_case
    m_ins = RHO_INS * math.pi * design["d_g"] * design["l_g"] * design["t_ins"]
    # spherical LOX tank, 90% usable, proof-sized
    v_lox = m_lox / RHO_LOX / 0.90
    r_tank = (3 * v_lox / (4 * math.pi)) ** (1.0 / 3.0)
    p_feed = 1.25 * p_c_max
    t_tank = max(J_PROOF * p_feed * r_tank / (2 * yield_6061(T_AMB) / J_MAT), 0.6e-3)
    m_tank = 1.20 * RHO_AL * 4 * math.pi * r_tank ** 2 * t_tank
    # injector, nozzle, valves/lines (lumped)
    a_inj = design["m_ox"] / (0.78 * math.sqrt(2 * RHO_LOX * 0.25 * p_c_max))
    m_inj = RHO_AL * a_inj * 0.02 + 0.05
    m_noz = 0.12 + 0.5 * (math.pi * design["d_t"] ** 2 / 4) * 1e4
    m_valves = 0.15
    m_loaded = m_fuel + m_lox + m_case + m_ins + m_tank + m_inj + m_noz + m_valves

    objectives = np.array([-Isp, m_loaded, T_case_max / 423.15, F_cov, of_range])

    # constraints g<=0, normalised to dimensionless relative violations so that
    # no single constraint dominates the aggregated violation by unit scale
    pe_pa_min = _min_pe_pa(design, bd)
    g = np.array([
        (285.0 - F_mean) / 300.0,        # 1 mean thrust >= 285
        (F_mean - 315.0) / 300.0,        # 2 mean thrust <= 315
        (3e6 - Pc.min()) / 5e6,          # 3 Pc >= 3 MPa (trajectory)
        (Pc.max() - 7e6) / 5e6,          # 4 Pc <= 7 MPa (trajectory)
        (1.25 - OF.min()) / 2.5,         # 5 O/F >= 1.25 (trajectory)
        (OF.max() - 4.50) / 2.5,         # 6 O/F <= 4.50 (trajectory)
        (20.0 - Gox.min()) / 300.0,      # 7 Gox >= 20 (trajectory)
        (Gox.max() - 850.0) / 300.0,     # 8 Gox <= 850 (trajectory)
        (0.002 - web) / 0.002,           # 9 final web >= 2 mm
        (T_case_max - 423.15) / 423.15,  # 10 peak case temp <= 423.15 K
        (0.40 - pe_pa_min) / 0.40,       # 11 exit-pressure ratio >= 0.40
    ])
    info = dict(F_mean=F_mean, Isp=Isp, p_c_max=p_c_max, of_mean=float(OF.mean()),
                web=web, T_case_max=T_case_max, m_loaded=m_loaded, F_cov=F_cov,
                of_range=of_range, t_case=t_case)
    return objectives, g, info


def _min_pe_pa(design, bd):
    """Minimum exit-pressure ratio over the burn (start and end states)."""
    vals = []
    for d_p in (design["d_p0"], bd["d_p_final"]):
        st = M.thrust_state(design["m_ox"], d_p, design["l_g"], design["d_t"],
                            design["eps"], design["w_al"], design["d_al"])
        vals.append(st["pe_pa"])
    return min(vals)
