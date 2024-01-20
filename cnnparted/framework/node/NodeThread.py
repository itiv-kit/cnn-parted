from framework.ModuleThreadInterface import ModuleThreadInterface
from .Timeloop import Timeloop
import os
import csv


class NodeThread(ModuleThreadInterface):
    def _eval(self) -> None:
        if not self.config:
            return

        if self.config.get("timeloop"):
            self._run_timeloop(self.config["timeloop"])
        else:
            self._run_generic(self.config)

    def _run_generic(self, config: dict) -> None:
        raise NotImplementedError

    def _run_timeloop(self, config: dict) -> None:
        runroot = self.runname + "_" + config["accelerator"]
        config["run_root"] = runroot
        fname_csv = runroot + "_convlayers.csv"

        if os.path.isfile(fname_csv):
            self.stats = self._read_layer_csv(fname_csv)
            return

        layers = self.ga.get_conv2d_layers()
        tl = Timeloop(config)
        tl.run(layers, self.progress)

        self.stats = tl.stats

        self._write_layer_csv(fname_csv)

    def _read_layer_csv(self, filename: str) -> dict:
        stats = {}

        with open(filename, 'r', newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                layer_name = row['Layer']
                stats[layer_name] = {}
                stats[layer_name]['latency'] = row['Latency [ms]']
                stats[layer_name]['energy'] = row['Energy [mJ]']

        return stats

    def _write_layer_csv(self, filename: str) -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            header = [
                "No.",
                "Layer",
                "Latency [ms]",
                "Energy [mJ]",
            ]
            writer.writerow(header)
            row_num = 1
            for l in self.stats.keys():
                if isinstance(self.stats[l], dict):
                    row = [
                        row_num,
                        l,
                        str(self.stats[l]["latency"]),
                        str(self.stats[l]["energy"]),
                    ]
                    writer.writerow(row)
                    row_num += 1


    def _remove_file(self,file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
