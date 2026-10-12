import os
import copy
from collections import defaultdict
import statistics

from loguru import logger
import numpy as np


from framework.optimizer.strategies.multi_tenancy import (MultiTenancyContext,
                                                          HeraldMultiTenancy, 
                                                          SaMultiTenancy, 
                                                          OptunaMultiTenancy,
                                                          PydeapMultiTenancy)

from framework.model.layer_tracker import LayerBindings
from framework.noc.noc import NocTopology
from framework.model.onnx_model import OnnxModel
from framework.model.layers.compute_layer import ComputationLayer
from framework.optimizer.optimizer import Optimizer
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.scheduling import SystemSchedule


class MultiTenancyPartitioning(Optimizer):
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer, global_conf: ConfigHelper, noc: NocTopology):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.ga = ga
        self.conf = global_conf
        self.system_result = node_stats
        self.noc = noc

        os.makedirs(self.plot_dir, exist_ok=True)

    def optimize(self):
        algorithm = MultiTenancyProblem(self.work_dir, self.system_result, self.ga, self.conf, self.noc)
        solutions, invalid_solutions = algorithm.minimize()
        return solutions, invalid_solutions


def _int_defaultdict():
    return defaultdict(int)

class MultiTenancyProblem():
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer, global_conf: ConfigHelper, noc: NocTopology):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.optuna_out_dir = os.path.join(work_dir, "optuna")
        self.node_stats = node_stats
        self.node_ids = node_stats.get_platform_ids()
        self.ga = ga
        self.noc = noc
        self.models = copy.deepcopy(ga.models)
        self.conf = global_conf

        # Parse execution graphs into LayerBindings and generate random initial binding
        network_info = defaultdict(_int_defaultdict)
        node_ids = self.node_stats.get_platform_ids()
        for network, model in ga.models.items():
            for layer in model.nodes:
                if isinstance(layer, ComputationLayer):
                    network_info[network][layer.layer_name] = np.random.choice(node_ids)

        self.layer_tracker = LayerBindings(network_info)
        self.num_nodes = self.layer_tracker.get_num_layers()



    def minimize(self) -> tuple[list[SystemSchedule], list[SystemSchedule]]:
        # Distribute workload between platforms
        logger.debug(f"Optimizing multitenancy problem with {self.conf.mt.strategy}")
        
        match self.conf.mt.strategy:
            case "optuna":
                strategy = OptunaMultiTenancy()
            case "sa":
                strategy = SaMultiTenancy()
            case "herald":
                strategy = HeraldMultiTenancy()
            case "deap":
                strategy = PydeapMultiTenancy()
            case _:
                raise RuntimeError(f"Unknown algorith: {self.conf.mt.strategy}")

        # Perform the optimization based on the specified strategy
        ctx = MultiTenancyContext(
            strategy=strategy,
            node_stats=self.node_stats,
            models=self.models,
            noc=self.noc,
            initial_tracker=self.layer_tracker,
            conf=self.conf,
        )
        solutions, invalid_solutions = ctx.run()

        if len(solutions) > 0:
            # Log some statistics
            system_runtimes: list[float] = []
            system_energy: list[float] = []
            for schedule in solutions:
                system_runtimes.append( max([sched.cur_time for sched in schedule.node_schedules.values()]) )
                system_energy.append(schedule.get_system_energy(self.node_stats))
            
            min_runtime_idx = system_runtimes.index(min(system_runtimes))
            min_runtime_latency = system_runtimes[min_runtime_idx]
            min_runtime_energy = system_energy[min_runtime_idx]
            min_energy_idx = system_energy.index(min(system_energy))
            min_energy_latency = system_runtimes[min_energy_idx]
            min_energy_energy = system_energy[min_energy_idx]
            logger.info(f"Configuration with minimum system runtime (ID: {min_runtime_idx}):\n\tRuntime: {min_runtime_latency}\n\tEnergy: {min_runtime_energy}")
            logger.info(f"Configuration with minimum system energy (ID: {min_energy_idx}):\n\tRuntime: {min_energy_latency}\n\tEnergy: {min_energy_energy}")
            logger.info(f"Median system runtime: {statistics.median(system_runtimes)}")

        else:
            logger.warning("No valid solutions for multi-tenancy scheduling were found")

        return solutions, invalid_solutions


