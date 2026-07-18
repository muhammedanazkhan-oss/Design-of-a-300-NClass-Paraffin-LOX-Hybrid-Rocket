"""
optimize.py -- the eight compared algorithms under an exact evaluation budget.

Global methods (via pymoo): NSGA-II, NSGA-III, SPEA2, DE.
Local refinement: a Scheduled Differential-Evolution Memetic (SDEM) phase that
combines coordinate (pattern) proposals with current-to-elite differential
proposals, the differential factor following a fixed schedule from 0.52 to 0.30.

Eight configurations (each spends exactly `budget` evaluations):
    nsga2, nsga3, spea2                      -- pure global
    de_only, pattern_only                    -- single-operator local ablations
    m_nsga3, m_spea2                         -- shared-local-search controls
    sdem                                     -- NSGA-II global + 720/480 SDEM local

'Scheduled', not 'adaptive': the factor schedule is indexed on evaluation count
with no feedback from search performance.
"""
from __future__ import annotations
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from .problem import HybridRocketProblem

POP = 60


def _global(name, seed, n_eval, dt, t_b):
    """Run a pure global algorithm for a capped number of evaluations."""
    problem = HybridRocketProblem(dt=dt, t_b=t_b)
    common = dict(pop_size=POP, sampling=LHS(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(prob=1.0 / 9.0, eta=20), eliminate_duplicates=True)
    if name == "nsga2":
        algo = NSGA2(**common)
    elif name == "spea2":
        algo = SPEA2(**common)
    elif name == "nsga3":
        ref = get_reference_directions("das-dennis", 5, n_partitions=6)
        algo = NSGA3(ref_dirs=ref, pop_size=POP, sampling=LHS(),
                     crossover=SBX(prob=0.9, eta=15),
                     mutation=PM(prob=1.0 / 9.0, eta=20), eliminate_duplicates=True)
    else:
        raise ValueError(name)
    n_gen = max(1, int(np.ceil(n_eval / POP)))
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False,
                   save_history=False)
    X = res.pop.get("X"); F = res.pop.get("F"); G = res.pop.get("G")
    return problem, X, F, G


def _dominates(f1, g1, f2, g2):
    """Constrained dominance: feasibility first, then Pareto dominance."""
    cv1 = np.sum(np.maximum(g1, 0)); cv2 = np.sum(np.maximum(g2, 0))
    if cv1 < cv2:
        return True
    if cv1 > cv2:
        return False
    return np.all(f1 <= f2) and np.any(f1 < f2)


def _sdem_local(problem, X, F, G, budget, rng, n_coord, n_de):
    """Local refinement phase. Mutates a working archive in place.

    Coordinate proposals perturb one variable at a time; DE proposals use a
    current-to-elite difference with a scheduled factor. Exactly n_coord + n_de
    evaluations are consumed (subject to the overall budget)."""
    xl, xu = problem.xl, problem.xu
    arch_X = list(X); arch_F = list(F); arch_G = list(G)
    total = n_coord + n_de
    used = 0
    ev = Evaluator()

    def evaluate_one(xq):
        xq = np.clip(xq, xl, xu)
        pop = Population.new(X=xq.reshape(1, -1))
        ev.eval(problem, pop)
        return pop.get("F")[0], pop.get("G")[0]

    order = rng.permutation(len(arch_X))
    ci = 0
    while used < total:
        base_i = int(order[ci % len(order)]); ci += 1
        xb = arch_X[base_i].copy()
        if used < n_coord:
            j = used % problem.n_var
            step = 0.05 * (xu[j] - xl[j]) * (1 if (used // problem.n_var) % 2 == 0 else -1)
            xq = xb.copy(); xq[j] += step
        else:
            frac = (used - n_coord) / max(n_de, 1)
            f_diff = 0.52 - (0.52 - 0.30) * frac        # scheduled factor
            a, b = rng.integers(0, len(arch_X), size=2)
            elite = arch_X[int(np.argmin([np.sum(f) for f in arch_F]))]
            xq = xb + f_diff * (elite - arch_X[a]) + f_diff * (arch_X[a] - arch_X[b])
        fq, gq = evaluate_one(xq)
        used += 1
        # accept if not dominated by any archive member
        dominated = any(_dominates(af, ag, fq, gq) for af, ag in zip(arch_F, arch_G))
        if not dominated:
            arch_X.append(np.clip(xq, xl, xu)); arch_F.append(fq); arch_G.append(gq)
    return np.array(arch_X), np.array(arch_F), np.array(arch_G)


def run_configuration(config, seed, budget=3000, dt=0.5, t_b=20.0):
    """Run one of the eight configurations for exactly `budget` evaluations.

    Returns (F, G) of the final population/archive.
    """
    rng = np.random.default_rng(seed)
    if config in ("nsga2", "nsga3", "spea2", "de"):
        problem, X, F, G = _global(config, seed, budget, dt, t_b)
        return F, G
    # memetic configurations: global phase then local refinement
    base = {"de_only": "nsga2", "pattern_only": "nsga2", "m_nsga3": "nsga3",
            "m_spea2": "spea2", "sdem": "nsga2"}[config]
    # 60 % of the budget on the global phase, 40 % on local refinement.
    # At budget 3000 this reproduces the 1800 global + 1200 local split, and the
    # SDEM 60/40 local split reproduces 720 coordinate + 480 differential trials.
    n_global = int(round(0.60 * budget))
    n_local = budget - n_global
    problem, X, F, G = _global(base, seed, n_global, dt, t_b)
    if config == "de_only":
        n_coord, n_de = 0, n_local
    elif config in ("pattern_only", "m_nsga3", "m_spea2"):
        n_coord, n_de = n_local, 0
    else:  # sdem
        n_coord = int(round(0.60 * n_local)); n_de = n_local - n_coord
    X2, F2, G2 = _sdem_local(problem, X, F, G, budget - n_global, rng, n_coord, n_de)
    return F2, G2


CONFIGS = ["nsga2", "nsga3", "spea2", "de_only", "pattern_only",
           "m_nsga3", "m_spea2", "sdem"]
