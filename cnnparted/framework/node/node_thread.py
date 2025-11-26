import os
import csv
import copy
import numpy as np

from framework.helpers.module_thread_interface import ModuleThreadInterface
from framework.node.timeloop import Timeloop
from framework.node.mnsim_interface import MNSIMInterface
from framework.node.generic_node import GenericNode
from framework.node.zigzag import Zigzag
from framework.node.node_evaluator import NodeResult
from framework.helpers.config import *


class NodeThread(ModuleThreadInterface):
    def eval_node(self) -> None:
        if not self.config:
            return

        # Check if design is DSE enabled
        top_k = -1
        metric = "edp"
        if self.config.dse is not None:
            # Check if dse_system_config is empty. If it is, raise an error here
            assert self.dse_system_config is not None, "Cannot specify DSE for a node if it is not also configured on system level"
        
            metric = self.dse_system_config.optimization
            top_k = int(self.dse_system_config.top_k)

        node_result_networks = []

        # Select which simulator should be used
        match self.config.evaluation.simulator:
            case "timeloop":
                simulator = Timeloop(self.config, self.dse_system_config)
            case "mnsim":
                simulator = MNSIMInterface(self.config)
            case "zigzag":
                simulator = Zigzag(self.config, self.runname)
            case _:
                simulator = GenericNode(self.config)

        # Evaluate all networks
        for network in self.ga.networks:

            # Check if some previous results are available
            fname_base = simulator.set_workdir(self.work_dir, network, self.id)
            if os.path.isfile(fname_base+".json"):
                stats = NodeResult.from_json(fname_base, network, self.config)
            else:
                # Perform the actual evaluation
                if self.acc_adaptor is None:
                    stats = simulator.run(network, self.ga.models[network])
                else:
                    stats = simulator.run_from_adaptor(network, self.ga.models[network], self.acc_adaptor)

                if self.save_results:
                    #stats.to_csv(fname_base)
                    stats.to_json(fname_base)

            node_result_networks.append(stats)
        
        self.stats = self._merge_network_results(node_result_networks)
        self.stats.prune_designs(top_k, metric)


    def _merge_network_results(self, network_results: list[NodeResult]):
        final_res = copy.deepcopy(network_results[0])

        if len(network_results) == 1:
            return final_res
        
        for network_result in network_results[1:]:
            for final_res_design, design in zip(final_res.designs, network_result.designs):
                nw = design.get_networks()[0]
                final_res_design.networks[nw] = copy.deepcopy(design.networks[nw])
        
        return final_res

    def _remove_file(self,file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
