import os
import torch.nn as nn
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

from typing import Callable

from .Optimizer import Optimizer
from .RobustnessProblem import RobustnessProblem


class RobustnessOptimizerCallback(Callback):
    def __init__(self, schedule: list, steps: list) -> None:
        super().__init__()
        assert len(schedule) == len(steps)
        self.schedule = schedule
        self.steps = steps

    def _update(self, algorithm: NSGA2):
        print(f"Reached generation {algorithm.n_iter}")
        if algorithm.n_iter in self.schedule:
            new_sample_limit = self.steps[self.schedule.index(algorithm.n_iter)]
            print(f"\tUpdating sample limit to {new_sample_limit}")
            algorithm.problem.update_sample_limit(new_sample_limit)


class RobustnessOptimizer(Optimizer):
    def __init__(self, run_name : str, model : nn.Module, accuracy_function : Callable, config : dict, progress : bool):
        self.fname_csv = run_name + "_" + "robustness.csv"

        self.sample_limit_schedule = config.get('robustness').get('sample_limit_generation_schedule')
        self.sample_limit_steps = config.get('robustness').get('sample_limit_steps')

        self.pop_size = config.get('robustness').get('pop_size')
        self.num_gen = config.get('robustness').get('num_gen')
        self.problem = RobustnessProblem(model, self.sample_limit_steps[0], config, accuracy_function, progress)

    def optimize(self):
        if not os.path.isfile(self.fname_csv):
            algorithm = NSGA2(
                pop_size=self.pop_size,
                n_offsprings=self.pop_size,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=5, repair=RoundingRepair()),
                mutation=PM(prob=0.9, eta=10, repair=RoundingRepair()),
                eliminate_duplicates=True)

            callback = RobustnessOptimizerCallback(self.sample_limit_schedule, self.sample_limit_steps)
            res = minimize( self.problem,
                            algorithm,
                            termination=get_termination('n_gen',self.num_gen),
                            seed=1,
                            save_history=False,
                            callback=callback,
                            verbose=False)

            if res.X.size == 0:
                print()
                print("### [RobustnessOptimizer] No valid bitwidth combination found! ###")
                print()
                quit()

            df = pd.DataFrame(res.X).replace(to_replace=range(0,len(self.problem.bits)), value=self.problem.bits)
            res.X = df.to_numpy()

            data = self._get_paretos_int(res)
            data = np.abs(data)
            data = np.delete(data, np.s_[0], axis=1) # delete G column

            df = pd.DataFrame(data)
            df.to_csv(self.fname_csv, header=False)
        else:
            df = pd.read_csv(self.fname_csv, header=None, index_col=0)
            data = df.to_numpy()

        constr = list(data[np.argmin(data, axis=0)[-3]])[:-3] # use configuration with min bit width sum
        constr_dict = {}
        for i, name in enumerate(self.problem.qmodel.explorable_module_names):
            constr_dict[name] = constr[int(i/2)]

        return constr_dict
