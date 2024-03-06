import os
import torch.nn as nn
import numpy as np
import pandas as pd

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
    def __init__(self, run_name : str, model : nn.Module, accuracy_function : Callable, config : dict, progress : bool):
        self.fname_csv = run_name + "_" + "robustness.csv"
        if os.path.isfile(self.fname_csv):
            self.run_optimize = False
        else:
            self.run_optimize = True
            self.pop_size = config.get('robustness').get('pop_size')
            self.num_gen = config.get('robustness').get('num_gen')
            self.problem = RobustnessProblem(model, config, accuracy_function, progress)

    def optimize(self):
        if self.run_optimize:
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

            df = pd.DataFrame(np.round(res.X)).replace(to_replace=range(0,len(self.problem.bits)), value=self.problem.bits)
            res.X = df.to_numpy()

            data = self._get_paretos_int(res)
            data = np.abs(data)

            df = pd.DataFrame(data)
            df.to_csv(self.fname_csv, header=False)
        else:
            df = pd.read_csv(self.fname_csv, header=None, index_col=0)
            data = df.to_numpy()

        return data[np.argmax(data, axis=0)[-2]] # return configuration with max accuracy
