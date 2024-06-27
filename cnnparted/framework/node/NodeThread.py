import os
import csv

from framework.ModuleThreadInterface import ModuleThreadInterface
from framework.node.Timeloop import Timeloop
from framework.node.MNSIMInterface import MNSIMInterface

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
        runroot = os.path.join(self.work_dir, "system_evaluation", config["accelerator"])
        config["run_root"] = runroot
        config["work_dir"] = self.work_dir
        fname_csv = os.path.join(self.work_dir, self.runname + "_" + config["accelerator"] + "_tl_layers.csv") #runroot + "_tl_layers.csv"

        if os.path.isfile(fname_csv):
            self.stats["eval"] = self._read_layer_csv(fname_csv)
            return

        layers = self.ga.get_timeloop_layers()
        tl = Timeloop(config)
        tl.run(layers, self.progress)

        self.stats["eval"] = tl.stats

        self._write_layer_csv(fname_csv)


    def _read_layer_csv(self, filename: str) -> dict:
        designs = []
        stats = {}

        with open(filename, 'r', newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            current_design_tag = "design0"
            for row in reader:
                if row["Design Tag"] != current_design_tag:
                    current_design_tag = row["Design Tag"]
                    designs.append(stats)
                    stats = {}
                layer_name = row['Layer']
                stats[layer_name] = {}
                stats[layer_name]['latency'] = row['Latency [ms]']
                stats[layer_name]['energy'] = row['Energy [mJ]']
                stats[layer_name]['area'] = row['Area [mm2]']

        return stats

    def _write_layer_csv(self, filename: str) -> None:
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
            for i, design in enumerate(self.stats["eval"]):
                for layer in design.keys():
                    if isinstance(design[layer], dict):
                        row = [
                            row_num,
                            f"design{i}",
                            layer,
                            str(design[layer]["latency"]),
                            str(design[layer]["energy"]),
                            str(design[layer]["area"])
                        ]
                        writer.writerow(row)
                        row_num += 1


    def _remove_file(self,file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
