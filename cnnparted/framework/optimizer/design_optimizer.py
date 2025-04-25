import os

import numpy as np
from numpy.random import default_rng

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from framework.optimizer.optimizer import Optimizer
from framework.optimizer.design_problem import DesignProblem, ACCELERATOR_CONFIG_MAP, N_VAR_ACC

# This modules does the following:
#   - Instatiate GesignOptimizerGaProblem
#   - use NSGA2 to optimize
class DesignOptimizer(Optimizer):

    def __init__(self, node_components, problem, algorithm: str):
        self.num_gen = 4
        self.pop_size = 4

        self.node_components = node_components
        
        self.problem = problem
        self.algorithm = algorithm

    def optimize(self):

        sorts = self._optimize_single()
        breakpoint()
        return sorts



    def _gen_initial_x(self):
        samples = []
        rng = default_rng()

        while(len(samples) < self.pop_size):
            for (node, constr) in zip(self.node_components, self.problem.node_constraints):
                sys_cfg = []
                if "dse" in node:
                    xl, xu = constr[0], constr[1]
                    acc_cfg = rng.integers(low=xl, high=[x+1 for x in xu])
                    sys_cfg.append(acc_cfg)
            sys_cfg = np.array(sys_cfg).flatten().tolist()

            if sys_cfg not in samples:
                samples.append(sys_cfg)
        
        return np.array(samples)


    def _optimize_single(self):
        problem = self.problem

        initial_x = self._gen_initial_x()

        pymoo_algorithms = ["nsga2"]
        pydeap_algorithms = []
        rl_algorithms = []

        if self.algorithm == "nsga2":
            algorithm = NSGA2(
                pop_size=self.pop_size,
                n_offsprings=self.pop_size,
                sampling=initial_x,
                crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
                mutation=PM(prob=0.9, eta=10, repair=RoundingRepair()),
                eliminate_duplicates=True
            )
        else:
            raise RuntimeError("Invalid algorithm {self.algorithm}")

        if self.algorithm in pymoo_algorithms:
            res = minimize(problem,
                        algorithm,
                        termination=get_termination('n_gen', self.num_gen),
                        seed=1,
                        save_history=True,
                        verbose=False
                        )
            data = self._get_paretos_int(res)
        return data
