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
from ..link.Link import Link


class NSGA2_Optimizer(Optimizer):
    def __init__(self, ga : GraphAnalyzer, nodeStats : dict, link_components : list, progress : bool) -> None:
        self.lgraph = ga.graph
        self.schedules = ga.schedules
        self.nodeStats = nodeStats
        self.link_confs = link_components
        self.progress = progress
        nodes = len(ga.schedules[0])

        self.num_gen = self.pop_size = 1
        if len(nodeStats.keys()) > 1:
            self.num_gen = 100 * nodes
            self.pop_size = 50 if nodes > 100 else nodes//2 if nodes > 30 else 15 if nodes > 20 else nodes


    def optimize(self, optimization_objectives):
        sorts = Parallel(n_jobs=1, backend="multiprocessing")(
            delayed(self._optimize_single)(s, self.lgraph, self.link_confs)
            for s in tqdm.tqdm(self.schedules, "Optimizer", disable=(not self.progress))
        )

        np.set_printoptions(precision=2)
        for s in sorts:
            for p in s:
                if p[-1]:
                    print(p)


    def _optimize_single(self, schedule : list, lgraph : LayersGraph, link_confs : list):
        problem = PartitioningProblem(self.nodeStats, schedule, lgraph, link_confs)

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
                        save_history=True,
                        verbose=False)

        X = np.round(res.X)
        F = np.abs(res.F)

        data = []
        for i in range(0, len(X)):
            data.append(np.append(X[i], F[i]))

        for h in res.history:
            for ind in h.pop:
                if ind.get("G") > 0:
                    continue
                data.append(np.append(np.round(ind.get("X").tolist()), ind.get("F").tolist()))
        data = np.unique(data, axis=0)

        x_len = len(X[0])
        comp_hist = np.delete(data, np.s_[0:x_len], axis=1)
        paretos = self._is_pareto_efficient(comp_hist)
        paretos = np.expand_dims(paretos, 1)
        data = np.hstack([data, paretos])

        return data


    # https://stackoverflow.com/a/40239615
    def _is_pareto_efficient(self, costs : np.ndarray, return_mask : bool = True) -> np.ndarray:
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


class PartitioningProblem(ElementwiseProblem):
    def __init__(self, nodeStats : dict, schedule : list, lgraph : LayersGraph, link_confs : list):
        self.nodeStats = nodeStats
        self.num_acc = len(nodeStats)
        self.num_pp = self.num_acc - 1

        n_var = self.num_pp * 2 + 1 ## Number of max. partitioning points + device ID

        self.schedule = schedule
        self.num_layers = len(schedule)

        self.lgraph = lgraph

        self.links = []
        for link_conf in link_confs:
            self.links.append(Link(link_conf))

        xu_pp = np.empty(self.num_pp)
        xu_pp.fill(self.num_layers + 0.49)
        xu_acc = np.empty(self.num_pp + 1)
        xu_acc.fill(self.num_acc + 0.49)

        xu = np.append(xu_pp, xu_acc)

        super().__init__(n_var=n_var,
                         n_obj=3,  ## latency and energy, add throughput?
                         n_constr=1,
                         xl=0.5,
                         xu=xu)


    def _evaluate(self, x, out, *args, **kwargs):
        p = []
        for i in x:
            p.append(int(np.round(i)))

        latency = 0.0
        energy = 0.0
        throughput = 0.0
        bandwidth = np.full((self.num_pp), np.inf)

        if not np.array_equal(np.sort(p[:self.num_pp-1]), p[:self.num_pp-1]):
            out["G"] = x[0]
        elif np.unique(p[self.num_pp:]).size != np.asarray(p[self.num_pp:]).size:
            out["G"] = x[0]
        else:
            out["G"] = -x[0]

            l_pp = []
            e_pp = []
            th_pp = []
            successors = []
            successors.append(self.schedule[0])

            i = -1
            last_pp = 0
            for i, pp in enumerate(p[0:self.num_pp], self.num_pp):
                self._eval_partition(p[i], last_pp, pp, l_pp, e_pp, th_pp, successors)
                last_pp = pp

                # evaluate link
                link_l, link_e, bandwidth[i-1] = self._get_link_metrics(i-1, successors)
                l_pp.append(link_l)
                e_pp.append(link_e)
                th_pp.append(self._zero_division(1.0, link_l))

            self._eval_partition(p[i+1], last_pp, self.num_layers, l_pp, e_pp, th_pp, successors)

            latency = sum(l_pp)
            energy = sum(e_pp)
            throughput = min(th_pp) * -1

        out["F"] = [latency, energy, throughput] + list(bandwidth)


    def _eval_partition(self, acc : int, last_pp : int, pp : int, l_pp : list, e_pp : list, th_pp : list, successors : list) -> None:
        acc -= 1
        acc_latency = 0.0
        acc_energy = 0.0
        for j in range(last_pp + 1, pp + 1):
            acc_latency += self._get_layer_latency(acc, j)
            acc_energy += self._get_layer_energy(acc, j)

            layer = self.schedule[j-1]
            while layer in successors: successors.remove(layer)
            successors += [s for s in self.lgraph.get_successors(layer)]

        l_pp.append(acc_latency)
        e_pp.append(acc_energy)
        th_pp.append(self._zero_division(1.0, acc_latency))


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

    def _get_link_metrics(self, link_idx : int, successors : list) -> (float, float, float):
        if len(self.links) == 0:
            return 0, 0, 0

        layers = []
        for layer in np.unique(successors):
            layers += list(self.lgraph.get_Graph().predecessors(layer))
        layers = np.unique([layer for layer in layers if layer not in successors])

        data_sizes = [np.prod(layer["output_size"]) for layer in self.lgraph.model_tree if layer.get("name") in layers]

        if len(self.links) == 1:
            return self.links[0].eval(np.sum(data_sizes))
        else:
            return self.links[link_idx].eval(np.sum(data_sizes))

