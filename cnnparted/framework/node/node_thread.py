import os
import csv
import copy
import numpy as np

from framework.module_thread_interface import ModuleThreadInterface
from framework.node.timeloop import Timeloop
from framework.node.mnsim_interface import MNSIMInterface
from framework.node.generic_node import GenericNode
from framework.node.zigzag import Zigzag
from framework.node.node_evaluator import NodeResult


class NodeThread(ModuleThreadInterface):
    def eval_node(self) -> None:
        if not self.config:
            return

        # Check if design is DSE enabled
        top_k = -1
        metric = "edp"
        if "dse" in self.config:
            # Check if dse_system_config is empty. If it is, raise an error here
            assert self.dse_system_config, "Cannot specify DSE for a node if it is not also configured on system level"
        
            metric = self.dse_system_config.get("optimization", "edp")
            top_k = int(self.dse_system_config.get("top_k", -1))

        node_result_networks = []

        # Evaluate all networks
        for network in self.ga.networks:

            # Select which simulator should be used
            match self.config["evaluation"]["simulator"]:
                case "timeloop":
                    layers = self.ga.get_timeloop_layers(network)
                    simulator = Timeloop(self.config, self.dse_system_config)
                case "mnsim":
                    layers = self.ga.get_mnsim_layers(network)
                    simulator = MNSIMInterface(self.config, self.ga.workloads[network].input_shape)
                case "zigzag":
                    layers = []
                    simulator = Zigzag(self.config, self.runname)
                case _:
                    layers = []
                    simulator = GenericNode(self.config)


            # Check if some previous results are available
            fname_base = simulator.set_workdir(self.work_dir, network, self.id)
            if os.path.isfile(fname_base+".csv"):
                #stats_j = NodeResult.from_json(fname_base, network, self.config)
                stats = NodeResult.from_csv(fname_base, network, self.config)
            else:
                # Perform the actual evaluation
                if self.acc_adaptor is None:
                    stats = simulator.run(network, layers)
                else:
                    stats = simulator.run_from_adaptor(network, layers, self.acc_adaptor)

                if self.save_results:
                    stats.to_csv(fname_base)
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
