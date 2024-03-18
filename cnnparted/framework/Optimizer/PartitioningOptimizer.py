import os
import numpy as np
import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from joblib import Parallel, delayed

from .Optimizer import Optimizer
from .PartitioningProblem import PartitioningProblem
from framework import GraphAnalyzer


class PartitioningOptimizer(Optimizer):
    def __init__(self, ga : GraphAnalyzer, num_pp : int, nodeStats : dict, link_components : list, progress : bool) -> None:
        self.run_name = ga.run_name
        self.schedules = ga.schedules
        self.num_pp = num_pp
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
            self.pop_size = 50

        self.results = {}

    def _set_layer_params(self, ga : GraphAnalyzer) -> dict:
        params = {}
        for layer in ga.get_conv2d_layers():
            params[layer['name']] = layer['conv_params']['weights']
        for layer in ga.get_gemm_layers():
            params[layer['name']] = layer['gemm_params']['weights']

        return params

    def optimize(self, q_constr : dict, fixed_sys : bool, acc_once : bool, opt : str, num_jobs : int) -> dict:
        all_paretos = []
        non_optimals = []

        fname_p_npy = self.run_name + "_" + "paretos.npy"
        fname_n_npy = self.run_name + "_" + "non_optimals.npy"
        if os.path.isfile(fname_p_npy) and os.path.isfile(fname_n_npy):
            all_paretos = list(np.load(fname_p_npy))
            non_optimals = list(np.load(fname_n_npy))
        else:
            sorts = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
                delayed(self._optimize_single)(self.num_pp, s, q_constr, fixed_sys, acc_once)
                for s in tqdm.tqdm(self.schedules, "Optimizer", disable=(not self.progress))
            )


            for i, sort in enumerate(sorts):
                for res in sort:
                    if res[-1]:
                        all_paretos.append(np.insert(res, 0, i)[:-1])
                    else:
                        non_optimals.append(np.insert(res, 0, i)[:-1])

            np.save(fname_p_npy, all_paretos)
            np.save(fname_n_npy, non_optimals)

        x_len = (self.num_pp) * 2 + 1
        comp_paretos = np.delete(all_paretos, np.s_[0:x_len+1], axis=1)

        if opt == 'edp':
            comp_paretos = self._pareto_edp(comp_paretos)
        else:
            comp_paretos = self._pareto_all(comp_paretos)

        all_paretos = np.hstack([all_paretos, np.expand_dims(self._is_pareto_efficient(comp_paretos), 1)])

        self.results["nondom"] = []
        self.results["dom"] = list(np.abs(non_optimals))
        for res in np.abs(all_paretos):
            if res[-1]:
                self.results["nondom"].append(res[:-1])
            else:
                self.results["dom"].append(res[:-1])

        return self.results

    def _pareto_edp(self, comp_paretos : np.ndarray) -> np.ndarray:
        comp_paretos = np.delete(comp_paretos, np.s_[2:], axis=1)
        comp_paretos = np.hstack([comp_paretos, np.expand_dims(np.prod(comp_paretos, axis=1), 1)])
        return comp_paretos

    def _pareto_all(self, comp_paretos : np.ndarray) -> np.ndarray:
        # TODO: Test
        bw_sums = np.sum(np.delete(np.delete(comp_paretos, np.s_[:self.num_pp-2], axis=1), np.s_[-self.num_pp:], axis=1), axis=1) # calc sum of bandwidths
        comp_paretos = np.delete(comp_paretos, np.s_[-(self.num_pp*2-1):], axis=1) # memories not relevant for finding pareto points
        comp_paretos = np.hstack([comp_paretos, np.expand_dims(bw_sums, 1)])

    def _optimize_single(self, num_pp : int, schedule : list, q_constr : dict, fixed_sys : bool, acc_once : bool) -> list:
        problem = PartitioningProblem(num_pp, self.nodeStats, schedule, q_constr, fixed_sys, acc_once, self.layer_dict, self.layer_params, self.link_confs)

        initial_x = np.concatenate((np.arange(1, num_pp+1), np.arange(1, num_pp+2) % len(self.nodeStats) + 1))
        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.pop_size,
            sampling=initial_x,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True)

        res = minimize( problem,
                        algorithm,
                        termination=get_termination('n_gen',self.num_gen),
                        seed=1,
                        save_history=True,
                        verbose=False)

        data = self._get_paretos_int(res)
        return data
