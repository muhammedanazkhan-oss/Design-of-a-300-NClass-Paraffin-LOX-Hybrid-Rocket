# results/

Generated outputs land here.

- `verification.json` (shipped) — deterministic; regenerate with
  `python scripts/run_verification.py` and diff to confirm reproduction.
- `uncertainty.json` — created by `python scripts/run_uncertainty.py`.
- `optimization.json`, `per_seed_indicators.csv` — created by
  `python scripts/run_optimization.py` (several hours at full budget; use `--quick`).
