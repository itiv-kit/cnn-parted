import os
import numpy as np
from numpy.random import default_rng
import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from joblib import Parallel, delayed

from framework.optimizer.optimizer import Optimizer
from framework.optimizer.partitioning_problem import PartitioningProblem
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult


class PartitioningOptimizer(Optimizer):
    def __init__(self, ga : GraphAnalyzer, num_pp : int, node_stats : SystemResult, link_components : list, progress : bool) -> None:
        node_stats = node_stats.to_dict()
        self.work_dir = ga.work_dir
        self.run_name = ga.run_name
        assert len(ga.networks) == 1, "PartitioningOptmizer does not support evaluation of multiple neural networks"
        network = ga.networks[0]
        self.network = network
        self.schedules = ga.schedules[network]
        self.num_pp = num_pp
        self.node_stats = node_stats
        self.link_confs = link_components
        self.progress = progress
        nodes = len(ga.schedules[network][0])

        self.layer_dict = {}
        for l in self.schedules[0]:
            self.layer_dict[l] = {}
            self.layer_dict[l]["predecessors"] = list(ga.graphs[network].get_graph().predecessors(l))
            self.layer_dict[l]["successors"] = [s for s in ga.graphs[network].get_successors(l)]
            self.layer_dict[l]["output_size"] = ga.graphs[network].output_sizes[l]

        self.layer_params = self._set_layer_params(ga)

        self.num_gen = self.pop_size = 2
        if len(node_stats.keys()) > 1 and num_pp > 0:
            self.num_gen = 50 * nodes
            self.pop_size = 50 if nodes>100 else nodes//2 if nodes>30 else 15 if nodes>20 else nodes

        self.results = {}
        self.optimizer_cfg = PartitioningOptConfig(len(self.node_stats), num_pp, len(self.schedules[0]))

    def _set_layer_params(self, ga : GraphAnalyzer) -> dict:
        params = {}
        network = ga.networks[0]
        for layer in ga.get_conv2d_layers(network):
            params[layer['name']] = layer['conv_params']['weights']
        for layer in ga.get_gemm_layers(network):
            params[layer['name']] = layer['gemm_params']['weights']

        return params

    def optimize(self, q_constr : dict, conf : dict, store_results: bool = True) -> tuple[int, int, dict]:
        fixed_sys = conf["general"].get('fixed_sys', False)
        acc_once = conf["general"].get('acc_once', False)
        opt = conf["general"].get('optimization')
        num_jobs = conf["general"].get('num_jobs')
        system_constraints = ConfigHelper.get_system_constraints(conf)
        max_num_platforms = conf["general"].get("max_num_platforms", None)

        all_paretos = []
        non_optimals = []
        g_len = self.optimizer_cfg.g_len #len(self.node_stats) + 1 + (self.num_pp + 1) * 2 + (self.num_pp + 1) * 2
        x_len = self.optimizer_cfg.x_len #(self.num_pp) * 2 + 1

        fname_p_npy = os.path.join(self.work_dir, self.run_name + "_" + "paretos.npy")
        fname_n_npy = os.path.join(self.work_dir, self.run_name + "_" + "non_optimals.npy")
        if os.path.isfile(fname_p_npy) and os.path.isfile(fname_n_npy):
            all_paretos = np.load(fname_p_npy)
            non_optimals = np.load(fname_n_npy)
        else:
            sorts = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
                delayed(self._optimize_single)(self.num_pp, s, q_constr, fixed_sys, acc_once, max_num_platforms, system_constraints)
                for s in tqdm.tqdm(self.schedules, "Optimizer", disable=(not self.progress))
            )

            num_platforms = len(self.node_stats)
            num_layers = len(self.schedules[0])
            nodeStatsIds = list(self.node_stats.keys())
            for i, sort in enumerate(sorts):
                if sort is not None:
                    for res in sort:
                        res[g_len:g_len+self.num_pp] = np.divide(res[g_len:g_len+self.num_pp], num_platforms).astype(int)
                        res[g_len+self.num_pp:g_len+x_len] = np.divide(res[g_len+self.num_pp:g_len+x_len], num_layers).astype(int)
                        
                        res[g_len:g_len+x_len] = np.floor(res[g_len:g_len+x_len]).astype(int) + 1
                        #res[g_len:g_len+self.num_pp] = np.floor(res[g_len:g_len+self.num_pp]).astype(int) + 1 #partitioning points
                        #res[g_len+self.num_pp:g_len+x_len] = [nodeStatsIds[int(p)] for p in res[g_len+self.num_pp:g_len+x_len]] #partitioning mapping
                        
                        if res[-1]:
                            all_paretos.append(np.insert(res, 0, i)[:-1]) # insert schedule ID and append to list
                        else:
                            non_optimals.append(np.insert(res, 0, i)[:-1]) # insert schedule ID and append to list

            all_paretos = np.unique(all_paretos, axis=0)
            non_optimals = np.unique(non_optimals, axis=0)

            if store_results:
                np.save(fname_p_npy, all_paretos)
                np.save(fname_n_npy, non_optimals)

        self.results["nondom"] = []
        self.results["dom"] = list(np.abs(non_optimals))
        if all_paretos.size:
            comp_paretos = np.delete(all_paretos, np.s_[0:g_len+x_len+1], axis=1)
            if opt == 'edp':
                comp_paretos = self._pareto_edp(comp_paretos)
            elif opt == 'ppa':
                comp_paretos = self._pareto_ppa(comp_paretos)

            all_paretos = np.hstack([all_paretos, np.expand_dims(self._is_pareto_efficient(comp_paretos), 1)])
            for res in np.abs(all_paretos):
                if res[-1]:
                    self.results["nondom"].append(res[:-1])
                else:
                    self.results["dom"].append(res[:-1])

        return g_len, x_len, self.results

    def _pareto_edp(self, comp_paretos : np.ndarray) -> np.ndarray:
        comp_paretos = np.delete(comp_paretos, np.s_[2:], axis=1) # only consider latency and energy
        comp_paretos = np.hstack([comp_paretos, np.expand_dims(np.prod(comp_paretos, axis=1), 1)])
        return comp_paretos

    def _pareto_ppa(self, comp_paretos : np.ndarray) -> np.ndarray:
        comp_paretos = np.delete(comp_paretos, np.s_[-2:], axis=1) # remove link metrics
        comp_paretos = np.delete(comp_paretos, np.s_[-2], axis=1) # remove throughput metric
        return comp_paretos

    def _gen_initial_x(self, num_layers, num_pp, fixed_sys, acc_once):
        num_platforms = len(self.node_stats)
        xu = num_platforms * num_layers - 1
        samples = []
        rng = default_rng()

        for i in range(num_platforms):
            pps = rng.integers(low=0, high=xu+1, size=num_pp)
            pps = np.sort(pps).tolist()
            accs = np.full(num_pp+1, i) * num_layers
            samples.append(pps + accs.tolist())

        while len(samples) < self.pop_size:
            pps = rng.integers(low=0, high=xu+1, size=num_pp)
            pps = np.sort(pps).tolist()
            accs = (rng.choice(num_platforms, size=num_pp+1, replace=not acc_once)) * num_layers
            if fixed_sys:
                accs = np.sort(accs)

            if pps + accs.tolist() not in samples or num_pp == 0:
                samples.append(pps + accs.tolist())
        return np.array(samples)

    def _optimize_single(self, num_pp : int, schedule : list, 
            q_constr : dict, fixed_sys : bool, acc_once : bool, max_num_platforms : int,
            system_constraints: dict) -> list:

        problem = PartitioningProblem(num_pp, self.node_stats, schedule, 
                    q_constr, fixed_sys, acc_once, max_num_platforms, self.layer_dict, 
                    self.layer_params, self.link_confs, system_constraints, self.optimizer_cfg, self.network)

        num_layers = len(schedule)
        initial_x = self._gen_initial_x(num_layers, num_pp, fixed_sys, acc_once)
        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=self.pop_size,
            sampling=initial_x,
            crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
            mutation=PM(prob=0.9, eta=10, repair=RoundingRepair()),
            eliminate_duplicates=True)

        res = minimize( problem,
                        algorithm,
                        termination=get_termination('n_gen',self.num_gen),
                        seed=1,
                        save_history=True,
                        verbose=False)

        data = self._get_paretos_int(res)
        return data
