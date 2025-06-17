import os
import csv
import numpy as np

from framework.module_thread_interface import ModuleThreadInterface
from framework.node.timeloop import Timeloop
from framework.node.mnsim_interface import MNSIMInterface
from framework.node.generic_node import GenericNode
from framework.node.zigzag import Zigzag
from framework.helpers.design_metrics import calc_metric
from framework.node.node_evaluator import LayerResult, NodeResult


class NodeThread(ModuleThreadInterface):
    def eval_node(self) -> None:
        if not self.config:
            return

        network = self.ga.networks[0] #TODO Multiple networks

        # Select which simulator should be used
        match self.config["evaluation"]["simulator"]:
            case "timeloop":
                layers = self.ga.get_timeloop_layers(network)
                simulator = Timeloop(self.config, self.dse_system_config)
            case "mnsim":
                layers = self.ga.get_mnsim_layers(network)
                simulator = MNSIMInterface(self.config, self.ga.workloads[network].input_size)
            case "zigzag":
                layers = []
                simulator = Zigzag(self.config)
            case _:
                layers = []
                simulator = GenericNode(self.config)

        # Check if design is DSE enabled
        top_k = -1
        metric = "edp"
        if "dse" in self.config:
            # Check if dse_system_config is empty. If it is, raise an error here
            assert self.dse_system_config, "Cannot specify DSE for a node if it is not also configured on system level"
        
            metric = self.dse_system_config.get("optimization", "edp")
            top_k = int(self.dse_system_config.get("top_k", -1))

        # Check if some previous results are available
        fname_csv = simulator.set_workdir(self.work_dir, network, self.id)
        if os.path.isfile(fname_csv):
            self.stats = NodeResult.from_csv(fname_csv, network, self.config)
            self.stats.prune_designs(top_k, metric)
            return

        # Perform the actual evaluation
        if self.acc_adaptor is None:
            self.stats = simulator.run(network, layers)
        else:
            self.stats = simulator.run_from_adaptor(network, layers, self.acc_adaptor)

        if self.save_results:
            self.stats.to_csv(fname_csv)
        
        self.stats.prune_designs(top_k, metric)


    def _remove_file(self,file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
