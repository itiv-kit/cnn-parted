import os
import csv
import numpy as np

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
        runroot = self.runname + "_" + str(self.id) + "_" + config["accelerator"]
        fname_csv = runroot + "_mnsim_layers.csv"

        layers = self.ga.get_mnsim_layers()
        mn = MNSIMInterface(layers, config, self.ga.input_size)

        if os.path.isfile(fname_csv):
            self.stats = self._read_layer_csv(fname_csv)
        else:
            mn.run()
            self.stats["eval"] = [mn.stats]
            self._write_layer_csv(fname_csv, stats=mn.stats)

        self.stats['bits'] = mn.pim_realADCbit()

    def _run_timeloop(self, config: dict) -> None:
        runroot = os.path.join(self.work_dir, "system_evaluation", str(self.id)+"_"+config["accelerator"])
        config["run_root"] = runroot
        config["work_dir"] = self.work_dir
        fname_csv = os.path.join(self.work_dir, self.runname + "_" + str(self.id) + "_" + config["accelerator"] + "_tl_layers.csv")
        
        # Check if design is DSE enabled
        if dse_cfg := config.get("dse"):
            is_dse = True
            metric = dse_cfg.get("optimization", "edap")
            top_k = int(dse_cfg.get("top_k", 2))
        else:
            is_dse = False
            metric= "edap"
            top_k = 1

        if os.path.isfile(fname_csv):
            read_stats = self._read_layer_csv(fname_csv)
            pruned_stats = self._prune_accelerator_designs(read_stats, top_k, metric, is_dse)
            self.stats["eval"] =  pruned_stats
            return

        layers = self.ga.get_timeloop_layers()
        tl = Timeloop(config)
        tl.run(layers, self.progress)

        self._write_layer_csv(fname_csv, tl.stats)

        pruned_stats = self._prune_accelerator_designs(tl.stats, top_k, metric, is_dse)

        # Prune accelerator design space 
        self.stats["eval"] = pruned_stats


    def _prune_accelerator_designs(self, stats: dict[str, dict], top_k: int, metric: str, is_dse: bool):
        # If there are less designs than top_k simply return the given list
        if len(stats) <= top_k or not is_dse:
            return stats

        # The metric_per_design array has this structure, with
        # every cell holding EAP, EDP or some other metric:
        #  x | l0 | l1 | l2 | l3 |
        # ------------------------
        # d0 | ...| ...| ...| ...|
        # d1 | ...| ...| ...| ...|
        metric_per_design = []
        energy_per_design = []
        latency_per_design = []
        area_per_design = []

        for tag, design in stats.items():
            #tag = design["tag"]
            layers = design["layers"]
            energy_per_layer = []
            latency_per_layer = []
            for name, layer in layers.items():
                energy_per_layer.append(layer["energy"])    
                latency_per_layer.append(layer["latency"])    

            energy_per_design.append(energy_per_layer)
            latency_per_design.append(latency_per_layer)
            area_per_design.append([layer["area"]])

        metric_per_design = calc_metric(np.array(energy_per_design), np.array(latency_per_design), np.array(area_per_design), metric, reduction=False)

        # Now, we need to find the top_k designs per layer
        design_candidates = []
        for col in metric_per_design.T:
            metric_for_layer = col.copy()
            metric_for_layer = np.argsort(metric_for_layer)

            for i in metric_for_layer[0:top_k]:
                design_candidates.append(f"design_{i}")

        design_candidates = np.unique(design_candidates) 

        # Remove all designs that have not been found to be suitable design candidates
        #pruned_stats = []
        #for design in stats:
        #    if design["tag"] in design_candidates: 
        #       tag = design["tag"] 
        #       layers = design["layers"]
        #       #arch_config = design["arch_config"]
        #       pruned_stats.append({"tag": tag, "layers": layers})
        
        pruned_stats = {tag: results for tag, results in stats.items() if tag in design_candidates}

        return pruned_stats

    def _apply_platform_constraints(self, stats: dict[str, dict], constraints: dict):
        max_energy = constraints.get("energy", np.inf)
        max_latency = constraints.get("latency", np.inf)
        max_area = constraints.get("area", np.inf)

        energy_per_design = []
        latency_per_design = []
        area_per_design = []
        for tag, design in stats.items():
            layers = design["layers"]
            energy_per_layer = []
            latency_per_layer = []
            for name, layer in layers.items():
                energy_per_layer.append(layer["energy"])    
                latency_per_layer.append(layer["latency"])    

            energy_per_design.append(energy_per_layer)
            latency_per_design.append(latency_per_layer)
            area_per_design.append([layer["area"]])

        energy_per_design = np.array(energy_per_design)
        latency_per_design = np.array(latency_per_design)
        area_per_design = np.array(area_per_design)
        
        total_energy = calc_metric(np.array(energy_per_design), np.array(latency_per_layer), np.array(area_per_design), "energy", reduction= True)
        total_latency = calc_metric(np.array(energy_per_design), np.array(latency_per_layer), np.array(area_per_design), "latency", reduction= True)
        total_area = area_per_design

        #constrained_stats = [design for idx, design in enumerate(stats) if (total_area[idx] <= max_area and total_latency[idx] <= max_latency and total_energy[idx] <= max_energy)]
        constrained_stats = {tag: design for idx, (tag, design) in stats.items() if (total_area[idx] <= max_area and total_latency[idx] <= max_latency and total_energy[idx] <= max_energy)}
        if not constrained_stats:
            raise ValueError("After applying constraints no designs remain!")

        return constrained_stats
        

    def _read_layer_csv(self, filename: str) -> dict:
        designs = {}
        stats = {}

        with open(filename, 'r', newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            current_design_tag = "design_0"
            stats["layers"] = {}
            for row in reader:
                if row["Design Tag"] != current_design_tag:
                    designs[current_design_tag] = stats
                    current_design_tag = row["Design Tag"]

                    stats = {}
                    stats["layers"] = {}

                layer_name = row['Layer']
                stats["layers"][layer_name] = {}
                stats["layers"][layer_name]['latency'] = float(row['Latency [ms]'])
                stats["layers"][layer_name]['energy'] = float(row['Energy [mJ]'])
                stats["layers"][layer_name]['area'] = float(row['Area [mm2]'])

        # Append stats for final design
        tag = row["Design Tag"]
        designs[tag] = stats
        return designs

    def _write_layer_csv(self, filename: str, stats: dict[str, dict] = None) -> None:
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
            for tag, design in stats.items():
                layers = design["layers"]
                for layer in layers.keys():
                    if isinstance(design["layers"][layer], dict):
                        row = [
                            row_num,
                            tag,
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
