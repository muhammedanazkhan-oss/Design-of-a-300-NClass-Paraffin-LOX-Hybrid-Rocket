"""Sanity tests for the reconstruction. Run: python -m pytest tests/ (or python tests/test_model.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from hybrid_recon import model as M, thermochem as tc, objectives, verification as V

DESIGN = dict(w_al=0.0, m_ox=0.085, d_p0=20.0e-3, l_g=350.0e-3, d_t=6.0e-3,
              eps=7.0, d_al=5.0, d_g=66.0e-3, t_ins=8.0e-3)

def test_calibration_anchor():
    r = tc.calibration_report()
    assert abs(r["cstar_at_anchor"] - r["cstar_anchor_target"]) < 1e-6

def test_cstar_ordering():
    # c* maximum must lie richer than the temperature maximum (pure fuel)
    of = np.linspace(1.3, 4.5, 400)
    of_cstar_max = of[tc.characteristic_velocity(of, 0.0).argmax()]
    of_temp_max = of[tc.chamber_temperature(of, 0.0).argmax()]
    assert of_cstar_max < of_temp_max

def test_pressure_closure_converges():
    st = M.solve_pressure(DESIGN["m_ox"], DESIGN["d_p0"], DESIGN["l_g"],
                          DESIGN["d_t"], DESIGN["w_al"], DESIGN["d_al"])
    assert st["iter_change"] < 1e-10 and st["eqn_residual"] < 1e-8

def test_rk4_order_near_four():
    orders = [o for _, o, _ in V.rk4_order(DESIGN)]
    assert 3.5 < np.mean(orders) < 4.5

def test_representative_feasible():
    _, g, _ = objectives.evaluate(DESIGN)
    assert g.max() <= 0.0

def test_determinism():
    _, g1, i1 = objectives.evaluate(DESIGN)
    _, g2, i2 = objectives.evaluate(DESIGN)
    assert abs(i1["Isp"] - i2["Isp"]) < 1e-9

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
