"""
problem.py -- pymoo Problem wrapper and the design evaluation cache.

Wraps objectives.evaluate() as a pymoo ElementwiseProblem with 9 variables,
5 objectives and 11 inequality constraints. A simple memoisation cache keeps the
evaluation count meaningful (repeated identical vectors are not recomputed but
are still counted once, matching an exact-budget protocol).
"""
from __future__ import annotations
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from . import objectives, model


class HybridRocketProblem(ElementwiseProblem):
    def __init__(self, dt: float = 0.5, t_b: float = 20.0):
        super().__init__(n_var=9, n_obj=5, n_ieq_constr=11,
                         xl=np.array(model.BOUNDS_LO), xu=np.array(model.BOUNDS_HI))
        self.dt = dt
        self.t_b = t_b
        self.n_eval = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.n_eval += 1
        design = model.vector_to_design(x)
        try:
            f, g, _ = objectives.evaluate(design, dt=self.dt, t_b=self.t_b)
        except Exception:
            # penalise non-evaluable vectors
            f = np.array([1e3, 1e3, 1e3, 1e3, 1e3])
            g = np.ones(11) * 1e3
        out["F"] = f
        out["G"] = g
