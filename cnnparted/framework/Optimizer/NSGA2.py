from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from .Optimizer import Optimizer
import numpy as np

import tqdm
from copy import deepcopy
from joblib import Parallel, delayed

from ..GraphAnalyzer import GraphAnalyzer
from ..link.Link import Link
from ..constants import NUM_JOBS


class NSGA2_Optimizer(Optimizer):
    def __init__(self, ga : GraphAnalyzer, nodeStats : dict, link_components : list, progress : bool) -> None:
        self.schedules = ga.schedules
        self.nodeStats = nodeStats
        self.link_confs = link_components
        self.progress = progress
        nodes = len(ga.schedules[0])

        self.layer_dict = {}
        for l in self.schedules[0]:
            self.layer_dict[l] = {}
            self.layer_dict[l]["predecessors"] = list(ga.graph.get_Graph().predecessors(l))
            self.layer_dict[l]["successors"] = [s for s in ga.graph.get_successors(l)]
            self.layer_dict[l]["output_size"] = ga.graph.output_sizes[l]

        self.layer_params = self._set_layer_params(ga)

        self.num_gen = self.pop_size = 1
        if len(nodeStats.keys()) > 1:
            self.num_gen = 100 * nodes
            self.pop_size = 50 if nodes > 100 else nodes//2 if nodes > 30 else 15 if nodes > 20 else nodes

    def _set_layer_params(self, ga : GraphAnalyzer) -> dict:
        params = {}
        for layer in ga.get_conv2d_layers():
            params[layer['name']] = layer['conv_params']['weights']
        for layer in ga.get_gemm_layers():
            params[layer['name']] = layer['gemm_params']['weights']

        return params

    def optimize(self, optimization_objectives):
        sorts = Parallel(n_jobs=NUM_JOBS, backend="multiprocessing")(
            delayed(self._optimize_single)(s)
            for s in tqdm.tqdm(self.schedules, "Optimizer", disable=(not self.progress))
        )

        np.set_printoptions(precision=2)
        print(sorts)
        for s in sorts:
            for p in s:
                if p[-1]:
                    print(p)

    def _optimize_single(self, schedule : list):
        problem = PartitioningProblem(self.nodeStats, schedule, self.layer_dict, self.layer_params, self.link_confs)

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
                data.append(np.append(np.round(ind.get("X").tolist()), np.abs(ind.get("F").tolist())))
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
    def __init__(self, nodeStats : dict, schedule : list, layer_dict : dict, layer_params : dict, link_confs : list):
        self.nodeStats = nodeStats
        self.num_acc = len(nodeStats)
        self.num_pp = self.num_acc - 1
        self.schedule = schedule
        self.num_layers = len(schedule)
        self.layer_dict = layer_dict
        self.layer_params = layer_params

        self.links = []
        for link_conf in link_confs:
            self.links.append(Link(link_conf))


        n_var = self.num_pp * 2 + 1 # Number of max. partitioning points + device ID
        n_obj = 3 + self.num_pp + self.num_acc # latency, energy, throughput + bandwidth + memory

        xu_pp = np.empty(self.num_pp)
        xu_pp.fill(self.num_layers + 0.49)
        xu_acc = np.empty(self.num_pp + 1)
        xu_acc.fill(self.num_acc + 0.49)

        xu = np.append(xu_pp, xu_acc)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
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
        mem = np.full((self.num_acc), np.inf)

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
                mem[i-1] = self._eval_partition(p[i], last_pp, pp, l_pp, e_pp, th_pp, successors)
                last_pp = pp

                # evaluate link
                link_l, link_e, bandwidth[i-1] = self._get_link_metrics(i-1, successors)
                l_pp.append(link_l)
                e_pp.append(link_e)
                th_pp.append(self._zero_division(1.0, link_l))

            mem[i] = self._eval_partition(p[i+1], last_pp, self.num_layers, l_pp, e_pp, th_pp, successors)

            latency = sum(l_pp)
            energy = sum(e_pp)
            throughput = min(th_pp) * -1

        out["F"] = [latency, energy, throughput] + list(bandwidth) + list(mem)

    def _eval_partition(self, acc : int, last_pp : int, pp : int, l_pp : list, e_pp : list, th_pp : list, successors : list) -> int:
        acc -= 1
        acc_latency = 0.0
        acc_energy = 0.0
        part_l_params = 0

        ls = {}
        dmem = []
        for j in range(last_pp + 1, pp + 1):
            layer = self.schedule[j-1]
            acc_latency += self._get_layer_latency(acc, layer)
            acc_energy += self._get_layer_energy(acc, layer)
            if layer in self.layer_params.keys():
                part_l_params += self.layer_params[layer]

            while layer in successors: successors.remove(layer)
            layer_successors = self.layer_dict[layer]["successors"]
            successors += layer_successors

            ifms = [] # Input Feature Maps
            afms = [] # Active Feature Maps
            for k in list(ls.keys()):
                v = ls[k]
                if layer in v:
                    ifms.append(np.prod(self.layer_dict[k]["output_size"]))
                    ls[k].remove(layer)
                elif v:
                    afms.append(np.prod(self.layer_dict[k]["output_size"]))
                else:
                    del ls[k]

            ofm = np.prod(self.layer_dict[layer]["output_size"]) # Output Feature Maps
            dmem.append(sum([ofm] + ifms + afms))   # Feature Map Memory Consumption per layer

            ls[layer] = deepcopy(layer_successors)

        l_pp.append(acc_latency)
        e_pp.append(acc_energy)
        th_pp.append(self._zero_division(1.0, acc_latency))
        return part_l_params + max(dmem, default=0) # mem evaluation

    def _get_layer_latency(self, acc : int, layer_name : str) -> float:
        acc = list(self.nodeStats.keys())[acc]
        if self.nodeStats[acc].get(layer_name):
            return float(self.nodeStats[acc][layer_name]['latency'])
        else:
            return 0

    def _get_layer_energy(self, acc : int, layer_name : str) -> float:
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
            layers += self.layer_dict[layer]["predecessors"]
        layers = np.unique([layer for layer in layers if layer not in successors])

        data_sizes = [np.prod(self.layer_dict[layer]["output_size"]) for layer in layers]

        if len(self.links) == 1:
            return self.links[0].eval(np.sum(data_sizes))
        else:
            return self.links[link_idx].eval(np.sum(data_sizes))

