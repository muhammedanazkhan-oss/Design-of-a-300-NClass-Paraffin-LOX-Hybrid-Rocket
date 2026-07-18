#!/usr/bin/env python3
"""Numerical verification: RK4 order, pressure residuals, quadrature comparison.

Usage:  python scripts/run_verification.py
Writes: results/verification.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hybrid_recon import verification as V, thermochem as tc

# a GENERIC example design vector (illustrative, not an optimised result)
DESIGN = dict(w_al=0.0, m_ox=0.085, d_p0=20.0e-3, l_g=350.0e-3, d_t=6.0e-3,
              eps=7.0, d_al=5.0, d_g=66.0e-3, t_ins=8.0e-3)

def main():
    out = {
        "thermochem_calibration": tc.calibration_report(),
        "rk4_order": [{"dt": dt, "order": o, "rel_err_pct": e}
                      for dt, o, e in V.rk4_order(DESIGN)],
        "pressure_residuals": V.residuals(DESIGN),
        "quadrature": V.quadrature_compare(DESIGN),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nWrote results/verification.json")

if __name__ == "__main__":
    main()
