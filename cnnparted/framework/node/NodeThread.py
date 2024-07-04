import os
import csv
import numpy as np
from typing import Dict

from framework.ModuleThreadInterface import ModuleThreadInterface
from framework.node.Timeloop import Timeloop
from framework.node.MNSIMInterface import MNSIMInterface
from framework.helpers.DesignMetrics import calc_metric

class NodeThread(ModuleThreadInterface):
    def _eval(self) -> None:
        if not self.config:
            return

        if self.config.get("timeloop"):
            self._run_timeloop(self.config["timeloop"])
            self.stats["type"] = 'tl'
        elif self.config.get("mnsim"):
            self._run_mnsim(self.config["mnsim"])
            self.stats["type"] = 'mnsim'
        else:
            self._run_generic(self.config)
            self.stats["type"] = 'generic'

        if 'bits' not in self.stats:
            self.stats["bits"] = self.config.get("bits") or 8

        self.stats["fault_rates"] = [float(i) for i in self.config.get("fault_rates") or [0.0, 0.0]]

    def _run_generic(self, config: dict) -> None:
        raise NotImplementedError

    def _run_mnsim(self, config: dict) -> None:
        runroot = self.runname + "_" + config["accelerator"]
        fname_csv = runroot + "_mnsim_layers.csv"

        layers = self.ga.get_mnsim_layers()
        mn = MNSIMInterface(layers, config, self.ga.input_size)

        if os.path.isfile(fname_csv):
            self.stats = self._read_layer_csv(fname_csv)
        else:
            mn.run()
            self.stats["eval"] = [mn.stats]
            self._write_layer_csv(fname_csv)

        self.stats['bits'] = mn.pim_realADCbit()

    def _run_timeloop(self, config: dict) -> None:
        #runroot = self.runname + "_" + config["accelerator"]
        runroot = os.path.join(self.work_dir, "system_evaluation", str(self.id)+"_"+config["accelerator"])
        config["run_root"] = runroot
        config["work_dir"] = self.work_dir
        fname_csv = os.path.join(self.work_dir, self.runname + "_" + config["accelerator"] + "_tl_layers.csv") #runroot + "_tl_layers.csv"

        if os.path.isfile(fname_csv):
            self.stats["eval"] = self._read_layer_csv(fname_csv)
            return

        layers = self.ga.get_timeloop_layers()
        tl = Timeloop(config)
        tl.run(layers, self.progress)

        write_stats = {}
        write_stats["eval"] = tl.stats
        self._write_layer_csv(fname_csv, stats=write_stats)

        # Prune accelerator design space 
        self.stats["eval"] = self._prune_accelerator_designs(tl.stats, 3, "edap")


    def _prune_accelerator_designs(self, stats: Dict, top_k: int, metric: str):
        # If there are less designs than top_k simply return the given list
        if len(stats) <= top_k:
            return stats

        # The metric_per_design array has this structure, with
        # every cell holding EAP, EDP or some other metric:
        #  x | l0 | l1 | l2 | l3 |
        # ------------------------
        # d0 | ...| ...| ...| ...|
        # d1 | ...| ...| ...| ...|
        metric_per_design = []
        labels = []

        for tag, design in stats.items():
            metric_per_layer = []
            layers = design["layers"]
            for key, layer in layers.items():
                metric_per_layer.append(calc_metric(layer, metric))

            labels.append(tag)
            metric_per_design.append(metric_per_layer)

        # Now, we need to find the top_k designs per layer
        design_candidates = []
        metric_per_design = np.array(metric_per_design)
        for col in metric_per_design.T:
            metric_for_layer = col.copy()
            metric_for_layer = np.argsort(metric_for_layer)

            for i in metric_for_layer[0:top_k]:
                design_candidates.append(f"design_{i}")

        design_candidates = np.unique(design_candidates) 

        # Remove all designs that have not been found to be suitable design candidates
        pruned_stats = {tag: design for tag, design in stats.items() if tag in design_candidates}
        return pruned_stats


    def _read_layer_csv(self, filename: str) -> dict:
        designs = []
        stats = {}

        with open(filename, 'r', newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            current_design_tag = "design_0"
            stats["layers"] = {}
            stats["tag"] = current_design_tag
            for row in reader:
                if row["Design Tag"] != current_design_tag:
                    current_design_tag = row["Design Tag"]
                    designs.append(stats)
                    stats["layers"] = {}
                    stats["tag"] = current_design_tag
                layer_name = row['Layer']
                stats["layers"][layer_name] = {}
                stats["layers"][layer_name]['latency'] = row['Latency [ms]']
                stats["layers"][layer_name]['energy'] = row['Energy [mJ]']
                stats["layers"][layer_name]['area'] = row['Area [mm2]']

        designs.append(stats)
        return designs

    def _write_layer_csv(self, filename: str, stats: Dict = None) -> None:
        if stats is None:
            stats = self.stats
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            header = [
                "No.",
                "Design Tag",
                "Layer",
                "Latency [ms]",
                "Energy [mJ]",
                'Area [mm2]'
            ]
            writer.writerow(header)
            row_num = 1
            for i, design in enumerate(stats["eval"]):
                layers = design["layers"]
                for layer in layers.keys():
                    if isinstance(design["layers"][layer], dict):
                        row = [
                            row_num,
                            design["tag"],
                            layer,
                            str(design["layers"][layer]["latency"]),
                            str(design["layers"][layer]["energy"]),
                            str(design["layers"][layer]["area"])
                        ]
                        writer.writerow(row)
                        row_num += 1


    def _remove_file(self,file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
