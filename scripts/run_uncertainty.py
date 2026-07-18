#!/usr/bin/env python3
"""Parametric Monte-Carlo stress test with per-constraint failure probabilities.

Usage:  python scripts/run_uncertainty.py [--quick]
Writes: results/uncertainty.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml
from hybrid_recon import uncertainty as U
from hybrid_recon.stats import wilson_interval

# THREE GENERIC EXAMPLE DESIGNS (illustrative inputs for the stress test, rounded
# and independent; NOT optimised results and NOT taken from any publication).
REPRESENTATIVES = {
    "example_high_thrust": dict(w_al=0.00, m_ox=0.085, d_p0=20.0e-3, l_g=350.0e-3, d_t=6.0e-3, eps=7.0, d_al=5.0,  d_g=66.0e-3, t_ins=8.0e-3),
    "example_large_port":  dict(w_al=0.00, m_ox=0.083, d_p0=27.0e-3, l_g=420.0e-3, d_t=6.1e-3, eps=6.8, d_al=25.0, d_g=62.0e-3, t_ins=10.0e-3),
    "example_metallized":  dict(w_al=0.03, m_ox=0.085, d_p0=30.0e-3, l_g=430.0e-3, d_t=6.1e-3, eps=5.0, d_al=5.0,  d_g=66.0e-3, t_ins=9.0e-3),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")))
    n = cfg["quick"]["n_samples"] if args.quick else cfg["uncertainty"]["n_samples"]
    seed = cfg["uncertainty"]["seed"]
    out = {}
    for name, design in REPRESENTATIVES.items():
        r = U.run(design, n_samples=n, seed=seed)
        lo, hi = wilson_interval(round(r["feasibility_pct"] / 100 * n), n)
        r["wilson95_pct"] = [lo, hi]
        out[name] = r
        print(f"{name:13s} feasibility {r['feasibility_pct']:.1f}% "
              f"(Wilson {lo:.1f}-{hi:.1f}%)")
        top = list(r["per_constraint_pct"].items())
        top = sorted(top, key=lambda kv: -kv[1])[:3]
        print("   dominant: " + ", ".join(f"{k} {v:.1f}%" for k, v in top if v > 0))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/uncertainty.json", "w"), indent=2)
    print("\nWrote results/uncertainty.json")

if __name__ == "__main__":
    main()
