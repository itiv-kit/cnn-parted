import os
import torch.nn as nn
import numpy as np
from numpy.random import default_rng
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

from framework.optimizer.Optimizer import Optimizer
from framework.optimizer.RobustnessProblem import RobustnessProblem


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
    def __init__(self, work_dir: str, run_name: str, model: nn.Module, accuracy_function: Callable, config: dict, device : str, progress: bool):
        self.fname_csv = os.path.join(work_dir, run_name + "_" + "robustness.csv")

        self.sample_limit_schedule = config.get('robustness').get('sample_limit_generation_schedule')
        self.sample_limit_steps = config.get('robustness').get('sample_limit_steps')

        self.pop_size = config.get('robustness').get('pop_size')
        self.num_gen = config.get('robustness').get('num_gen')
        self.problem = RobustnessProblem(model, self.sample_limit_steps[0], config, device, accuracy_function, progress)

        self.max_bits_idx = [len(config.get('robustness').get('bits')) - 2, len(config.get('robustness').get('bits')) - 1]

        self.delta = config.get('robustness').get('delta')

    def optimize(self):
        rng = default_rng(seed=42)
        n_var = self.problem.n_var
        init_x = [np.full(n_var, self.max_bits_idx[-1]), np.full(n_var, self.max_bits_idx[-2])]
        for i in range(self.pop_size-2):
            init_x.append(rng.choice(self.max_bits_idx, size=n_var))

        if not os.path.isfile(self.fname_csv):
            algorithm = NSGA2(
                pop_size=self.pop_size,
                n_offsprings=self.pop_size,
                sampling=np.array(init_x),
                crossover=SBX(prob=0.9, eta=5, repair=RoundingRepair()),
                mutation=PM(prob=0.9, eta=10, repair=RoundingRepair()),
                eliminate_duplicates=True)

            callback = RobustnessOptimizerCallback(self.sample_limit_schedule, self.sample_limit_steps)
            res = minimize(self.problem,
                           algorithm,
                           termination=get_termination('n_gen', self.num_gen),
                           seed=1,
                           save_history=False,
                           callback=callback,
                           verbose=False)

            if not isinstance(res.X, np.ndarray):
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

        best_idx = np.argmax(data, axis=0)[-2]
        lowest_accuracy = data[best_idx][-2] - self.delta

        sel_idx = 0
        sel_accuracy = data[best_idx][-2]
        for i, x in enumerate(data):
            if x[-2] >= lowest_accuracy and x[-2] < sel_accuracy:
                sel_idx = i
                sel_accuracy = x[-2]

        constr = data[sel_idx][:-3]
        constr_dict = {}
        for i, name in enumerate(self.problem.qmodel.explorable_module_names):
            constr_dict[name] = constr[int(i/2)]

        return constr_dict
