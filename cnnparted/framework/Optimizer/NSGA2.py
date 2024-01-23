from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import numpy as np
import tqdm
from joblib import Parallel, delayed


from .Optimizer import Optimizer
from .PartitioningProblem import PartitioningProblem
from ..GraphAnalyzer import GraphAnalyzer
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
        return Parallel(n_jobs=NUM_JOBS, backend="multiprocessing")(
            delayed(self._optimize_single)(s)
            for s in tqdm.tqdm(self.schedules, "Optimizer", disable=(not self.progress))
        )

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
