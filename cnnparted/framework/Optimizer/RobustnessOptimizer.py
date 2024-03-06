import torch.nn as nn
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.factory import get_termination

from typing import Callable

from .Optimizer import Optimizer
from .RobustnessProblem import RobustnessProblem


class RobustnessOptimizer(Optimizer):
    def __init__(self, model : nn.Module, accuracy_function : Callable, config : dict, progress : bool):
        self.pop_size = config.get('robustness').get('pop_size')
        self.num_gen = config.get('robustness').get('num_gen')
        self.problem = RobustnessProblem(model, config, accuracy_function, progress)

    def optimize(self):
        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=5),
            mutation=PM(prob=0.9, eta=10),
            eliminate_duplicates=True)

        res = minimize( self.problem,
                        algorithm,
                        termination=get_termination('n_gen',self.num_gen),
                        seed=1,
                        save_history=False,
                        verbose=False)

        data = self._get_paretos_int(res)
        return np.abs(data)
