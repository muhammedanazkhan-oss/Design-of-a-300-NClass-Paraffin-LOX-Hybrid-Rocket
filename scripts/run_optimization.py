#!/usr/bin/env python3
"""Eight-algorithm comparison under an exact evaluation budget.

Builds a held-out reference set from all eight configurations (independent
seeds), then for each of the paired test seeds runs every configuration,
thins and normalises the fronts, and computes HV and IGD+. Finally runs the
Friedman / Wilcoxon / Holm analysis of SDEM against every comparator.

Usage:  python scripts/run_optimization.py [--quick]
Writes: results/optimization.json, results/per_seed_indicators.csv
WARNING: the full protocol (8 configs x 20 seeds x 3000 evals + reference) is
computationally heavy (order of hours). Use --quick for a fast smoke test.
"""
import argparse, csv, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, yaml
from hybrid_recon.optimize import run_configuration, CONFIGS
from hybrid_recon import indicators as I
from hybrid_recon import stats as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")))
    q = cfg["quick"] if args.quick else cfg["optimization"]
    budget = q["budget"] if args.quick else cfg["optimization"]["budget"]
    seeds = q["seeds"] if args.quick else cfg["optimization"]["seeds"]
    ref_seeds = q["reference_seeds"] if args.quick else cfg["optimization"]["reference_seeds"]
    ref_budget = q["reference_budget"] if args.quick else cfg["optimization"]["reference_budget"]
    cap = cfg["optimization"]["indicator_cap"]
    hv_pt = cfg["optimization"]["hv_reference_point"]
    dt, t_b = cfg["burn"]["dt"], cfg["burn"]["t_b"]

    t0 = time.time()
    # ---- held-out reference set: all eight configs, independent seeds ----------
    print("Building held-out reference set (all eight configurations)...")
    ref_fronts = []
    for cf in CONFIGS:
        for s in ref_seeds:
            F, G = run_configuration(cf, seed=s, budget=ref_budget, dt=dt, t_b=t_b)
            ref_fronts.append(I.feasible_objectives(F, G))
    reference = I.build_reference(ref_fronts)
    if len(reference) == 0:
        print("No feasible reference points found; increase budget."); return
    ideal = reference.min(axis=0); nadir = reference.max(axis=0)
    ref_norm = I.normalise(reference, ideal, nadir)
    print(f"  reference points: {len(reference)}")

    # ---- paired test runs ------------------------------------------------------
    hv = {c: [] for c in CONFIGS}; igd = {c: [] for c in CONFIGS}
    rows = []
    for s in seeds:
        for cf in CONFIGS:
            F, G = run_configuration(cf, seed=s, budget=budget, dt=dt, t_b=t_b)
            feas = I.feasible_objectives(F, G)
            nd = I.non_dominated(feas)
            nd = I.farthest_point_thin(nd, cap) if len(nd) else nd
            fn = I.normalise(nd, ideal, nadir) if len(nd) else nd
            h, g = I.indicators(fn, ref_norm, hv_pt)
            hv[cf].append(h); igd[cf].append(g)
            rows.append(dict(seed=s, config=cf, hv=h, igd_plus=g, front=len(nd)))

    # ---- statistics: SDEM vs each comparator -----------------------------------
    hv_mat = np.array([hv[c] for c in CONFIGS]).T
    igd_mat = np.array([igd[c] for c in CONFIGS]).T
    chi_hv, p_hv = S.friedman(hv_mat)
    chi_igd, p_igd = S.friedman(igd_mat)
    comparators = [c for c in CONFIGS if c != "sdem"]
    raw_p = []; effect = []
    for c in comparators:
        p1, r1 = S.wilcoxon_one_sided(hv["sdem"], hv[c], "greater")
        p2, r2 = S.wilcoxon_one_sided(igd["sdem"], igd[c], "less")
        raw_p += [p1, p2]; effect += [("HV_vs_"+c, r1), ("IGD_vs_"+c, r2)]
    adj = S.holm(raw_p)

    summary = dict(
        note="Independent reconstruction. Values reproduce THIS package only.",
        budget=budget, n_seeds=len(seeds), reference_points=len(reference),
        friedman={"HV": {"chi2": chi_hv, "p": p_hv},
                  "IGD_plus": {"chi2": chi_igd, "p": p_igd}},
        medians={c: {"HV": float(np.median(hv[c])), "IGD_plus": float(np.median(igd[c]))}
                 for c in CONFIGS},
        sdem_vs_comparators=[
            {"test": effect[i][0], "holm_p": float(adj[i]), "rank_biserial": float(effect[i][1])}
            for i in range(len(adj))],
        runtime_s=round(time.time() - t0, 1),
    )
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open("results/optimization.json", "w"), indent=2)
    with open("results/per_seed_indicators.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "config", "hv", "igd_plus", "front"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary["medians"], indent=2))
    print(f"Friedman HV p={p_hv:.2e}  IGD+ p={p_igd:.2e}")
    print(f"Wrote results/optimization.json and per_seed_indicators.csv  ({summary['runtime_s']}s)")

if __name__ == "__main__":
    main()
