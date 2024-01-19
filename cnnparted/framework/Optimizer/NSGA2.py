from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling #IntegerRandomSampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from .Optimizer import Optimizer
import numpy as np

import tqdm
from joblib import Parallel, delayed

from ..GraphAnalyzer import GraphAnalyzer
from ..model.graph import LayersGraph

class NSGA2_Optimizer(Optimizer):
    def __init__(self, ga : GraphAnalyzer, nodeStats, link_components) -> None:
        nodes = len(ga.schedules[0])
        self.num_gen = 100 * nodes #10 time node size
        self.pop_size = 50 if nodes > 100 else nodes//2 if nodes > 30 else 15 if nodes > 20 else nodes

        self.lgraph = ga.graph
        self.schedules = ga.schedules
        self.nodeStats = nodeStats

        self.links = link_components

    def optimize(self, optimization_objectives):
        sorts = Parallel(n_jobs=1, backend="multiprocessing")(
            delayed(self._optimize_single)(s, self.lgraph)
            for s in tqdm.tqdm(self.schedules)
        )

        np.set_printoptions(precision=2)
        print(sorts)

    def _optimize_single(self, schedule : list, lgraph : LayersGraph):
        problem = PartitioningProblem(self.nodeStats, schedule, lgraph)

        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize( problem,
                        algorithm,
                        termination=get_termination('n_gen',self.num_gen),
                        seed=1,
                        save_history=False,
                        verbose=False)

        X = res.X
        F = res.F

        X_rounded=np.round(res.X)

        PP = []
        for i in range(0, len(X)):
            PP.append(np.append(X_rounded[i], F[i]))

        PP_unique = np.unique(PP, axis=0)

        return PP_unique

class PartitioningProblem(ElementwiseProblem):
    def __init__(self, nodeStats : dict, schedule : list, lgraph : LayersGraph):
        self.nodeStats = nodeStats
        self.num_acc = len(nodeStats)
        self.num_pp = self.num_acc - 1
        assert self.num_acc > 1, "Only one accelerator found"

        n_var = self.num_pp * 2 + 1 ## Number of max. partitioning points + device ID

        self.schedule = schedule
        self.num_layers = len(schedule)

        self.lgraph = lgraph

        xu_pp = np.empty(self.num_pp)
        xu_pp.fill(self.num_layers)
        xu_acc = np.empty(self.num_pp + 1)
        xu_acc.fill(self.num_acc)

        xu = np.append(xu_pp, xu_acc)

        super().__init__(n_var=n_var,
                         n_obj=3,  ## latency and energy, add throughput?
                         n_constr=1,
                         xl=1,
                         xu=xu)


    def _evaluate(self, x, out, *args, **kwargs):
        p = []
        for i in x:
            p.append(int(np.round(i)))

        latency = 0.0
        energy = 0.0
        throughput = 0.0

        if not np.array_equal(np.sort(p[:self.num_pp-1]), p[:self.num_pp-1]):
            out["G"] = x[0]
        elif np.unique(p[self.num_pp:]).size != np.asarray(p[self.num_pp:]).size:
            out["G"] = x[0]
        else:
            out["G"] = -x[0]

            last_pp = 0
            th_pp = []

            for i, pp in enumerate(p[0:self.num_pp], self.num_pp):
                successors = []
                successors.append(self.schedule[0])

                acc = p[i] - 1
                acc_latency = 0.0
                for j in range(last_pp + 1, pp + 1):
                    acc_latency += self._get_layer_latency(acc, j)
                    latency += self._get_layer_latency(acc, j)
                    energy += self._get_layer_energy(acc, j)

                    layer = self.schedule[j-1]
                    while layer in successors: successors.remove(layer)
                    successors += [s for s in self.lgraph.get_successors(layer)]

                th_pp.append(self._zero_division(1.0, acc_latency))
                last_pp = pp

                link_l, link_e = self._get_link_metrics(successors)
                th_pp.append(self._zero_division(1.0, link_l))
                latency += link_l
                energy += link_e

                if pp == p[self.num_pp - 1]:
                    acc = p[i+1] - 1
                    acc_latency = 0.0
                    for j in range(last_pp + 1, self.num_layers + 1):
                        acc_latency += self._get_layer_latency(acc, j)
                        latency += self._get_layer_latency(acc, j)
                        energy += self._get_layer_energy(acc, j)

                    th_pp.append(self._zero_division(1.0, acc_latency))

            throughput = min(th_pp) * -1

        out["F"] = [latency, energy, throughput]

    def _get_layer_latency(self, acc : int, idx : int) -> float:
        layer_name = self.schedule[idx-1]
        acc = list(self.nodeStats.keys())[acc]

        if self.nodeStats[acc].get(layer_name):
            return float(self.nodeStats[acc][layer_name]['latency'])
        else:
            return 0

    def _get_layer_energy(self, acc : int, idx : int) -> float:
        layer_name = self.schedule[idx-1]
        acc = list(self.nodeStats.keys())[acc]

        if self.nodeStats[acc].get(layer_name):
            return float(self.nodeStats[acc][layer_name]['energy'])
        else:
            return 0

    def _zero_division(self, a : float, b : float) -> float:
        return a / b if b else np.inf

    def _get_link_metrics(self, successors : list) -> (float, float):
        layers = []
        for layer in np.unique(successors):
            layers += list(self.lgraph.get_Graph().predecessors(layer))
        layers = np.unique([layer for layer in layers if layer not in successors])

        data_sizes = [np.prod(layer["output_size"]) for layer in self.lgraph.model_tree if layer.get("name") in layers]

        print(np.sum(data_sizes))

        return 0, 0
