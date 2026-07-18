#!/usr/bin/env python3
"""One-command reproduction of the full reconstruction pipeline.

Usage:  python scripts/run_all.py [--quick]
Runs verification, optimization, and uncertainty in sequence.
"""
import argparse, subprocess, sys, os
HERE = os.path.dirname(__file__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    extra = ["--quick"] if args.quick else []
    for script in ("run_verification.py", "run_optimization.py", "run_uncertainty.py"):
        print(f"\n{'='*70}\n  {script}\n{'='*70}")
        cmd = [sys.executable, os.path.join(HERE, script)]
        if script != "run_verification.py":
            cmd += extra
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
