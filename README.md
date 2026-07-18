# Hybrid Rocket Design Model — Independent Reconstruction

A self-contained, runnable re-implementation of a thermodynamically constrained,
transient, many-objective design model for a **300 N-class paraffin / liquid-oxygen
(optionally aluminized) hybrid rocket**, together with the optimization,
verification and uncertainty drivers.

---

## ⚠️ Read this first — what this package is, and is not

**This is an INDEPENDENT reconstruction built from a model's governing equations.
It is NOT the original authors' code, and it does NOT reproduce any specific
published table, figure, or numeric result.**

- It was written from the *equations* of the design model, not from the original
  simulation/optimization source, which was not available.
- The thermochemistry uses a **calibrated analytical surrogate** in place of a live
  Cantera equilibrium grid (the exact Cantera setup was not available). It preserves
  the correct thermodynamic *ordering* but returns *approximate* absolute values.
- Because the optimizer, random seeds, thermochemical grid and hardware sub-models
  are independent re-implementations, the numbers this code produces **differ** from
  any external publication. For example, this reconstruction yields a mean specific
  impulse near 258 s for its high-Isp representative, not a value read from any paper.
- What it **does** guarantee: it runs end-to-end and reproduces **its own** results
  **deterministically** for a fixed seed. That is the property this archive certifies.

If you need to reproduce a specific paper's tables, you need that paper's original
code. This package cannot substitute for it and does not claim to.

---

## What it contains

| Area | Module | What it does |
|------|--------|--------------|
| Thermochemistry | `src/hybrid_recon/thermochem.py` | Calibrated `c*`, `T_c`, `M`, `γ`, condensed/gas fractions; `c*` re-derived from fundamentals |
| Transient model | `src/hybrid_recon/model.py` | Regression, RK4 port growth, pressure closure, nozzle expansion, two-phase loss |
| Objectives | `src/hybrid_recon/objectives.py` | 5 objectives, 11 normalised constraints, two-node thermal, derated case, mass bookkeeping |
| Optimizer | `src/hybrid_recon/optimize.py` | 8 configurations incl. the Scheduled DE-Memetic (SDEM) local search |
| Indicators | `src/hybrid_recon/indicators.py` | Held-out reference set, farthest-point thinning, HV, IGD+ |
| Statistics | `src/hybrid_recon/stats.py` | Friedman, one-sided Wilcoxon, Holm, rank-biserial, Wilson intervals |
| Uncertainty | `src/hybrid_recon/uncertainty.py` | Monte-Carlo per-constraint failure decomposition |
| Verification | `src/hybrid_recon/verification.py` | RK4 order, both residual quantities, quadrature comparison |

Drivers live in `scripts/`; settings in `config/default.yaml`; tests in `tests/`.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Python 3.10+ is required. The only heavy dependency is `pymoo` (the multi-objective
optimizers and quality indicators).

## Running

```bash
# fast smoke test of the whole pipeline (minutes)
python scripts/run_all.py --quick

# individual stages
python scripts/run_verification.py            # RK4 order, residuals, quadrature
python scripts/run_uncertainty.py             # 2000-sample per-constraint stress test
python scripts/run_optimization.py            # 8-algorithm comparison (SLOW — see below)

# unit / sanity tests
python tests/test_model.py
```

Outputs are written to `results/` as JSON and CSV.

### Runtime warning

A single model evaluation costs roughly 25–30 ms. The **full** optimization protocol
is `8 configs × 20 seeds × 3000 evaluations` plus a held-out reference set, i.e. of
order 5×10⁵ evaluations, which takes **several hours** on one core. Use `--quick`
(reduced budget and seeds) to confirm the pipeline before committing to a full run.

---

## Reproducibility

- Every stochastic step is seeded; `config/default.yaml` lists the seeds.
- `run_verification.py` and `run_uncertainty.py` are deterministic for a fixed seed.
- `run_optimization.py` is deterministic per (config, seed) pair.
- `tests/test_model.py` asserts the calibration anchor, the `c*`/temperature
  ordering, closure convergence, RK4 order, feasibility, and determinism.

The environment used to produce the shipped `results/` files is recorded in
`ENVIRONMENT.txt`.

---

## Model scope and known simplifications

This is a **screening-tier** model. In particular:

- The two-phase nozzle loss folds the gas fraction into the `c*` relation and is
  **not** a full homogeneous two-phase choked-flow derivation; **aluminized results
  are indicative only**.
- The thermochemistry is a calibrated surrogate, not a live equilibrium solve; no
  grid-convergence or withheld-state validation is performed.
- The regression correlation is a constructed screening form, not a least-squares fit
  to a named dataset; transfer from gaseous- to liquid-oxygen operation is not modelled.
- The thermal model is two-node with lumped coefficients; the hardware/feed masses are
  conceptual estimates.

These are stated so the outputs are read as a screening front, not a hardware release.

---

## License

Code is released under the MIT License (`LICENSE`). See `CITATION.cff` for how to
cite this reconstruction.
