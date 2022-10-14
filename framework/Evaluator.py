import csv
import statistics
from collections import OrderedDict
from typing import Optional

from .DNNAnalyzer import DNNAnalyzer

class Evaluator():
    def __init__(self, dnn : DNNAnalyzer, sensorStats : dict, linkStats : dict, edgeStats : dict) -> None:
        self.dnn = dnn
        self.sensorStats = self._get_median(sensorStats)
        self.linkStats = self._get_median(linkStats)
        self.edgeStats = self._get_median(edgeStats)
        self._evaluate()

    def print_sim_time(self) -> None:
        print()
        print("Median Simulation Time")
        print("===========================================")
        print("DNN Analyzer:", self.dnn.stats['sim_time'], "s")
        print("Sensor Node: ", self.sensorStats['sim_time'], "s")
        print("Link:        ", self.linkStats['sim_time'],   "s")
        print("Edge Node:   ", self.edgeStats['sim_time'],   "s")
        print()

    def _get_median(self, stat : dict) -> None:
        avg_stats = {}
        for j in stat[0].keys():
            if isinstance(stat[0][j], dict):
                avg_stats[j] = {}
                for k in stat[0][j].keys():
                    if isinstance(stat[0][j][k], dict):
                        avg_stats[j][k] = {}
                        for m in stat[0][j][k].keys():
                            mdn = statistics.median([stat[n][j][k][m] for n in stat.keys()])
                            avg_stats[j][k][m] = mdn
                    else:
                        mdn = statistics.median([stat[n][j][k] for n in stat.keys()])
                        avg_stats[j][k] = mdn
            else:
                mdn = statistics.median([stat[n][j] for n in stat.keys()])
                avg_stats[j] = mdn

        return avg_stats

    def _evaluate(self) -> None:
        self.res = OrderedDict()

        for layer in self.dnn.partition_points:
            id = layer.get_layer_name(False, True)

            self.res[id] = {}
            self.res[id]['output_size'] = layer.output_size

            if id in self.sensorStats.keys():
                self.res[id]['sensor_latency'] = self.sensorStats[id]['latency']
                self.res[id]['sensor_energy'] = self.sensorStats[id]['energy']
            else:
                self.res[id]['sensor_latency'] = 0
                self.res[id]['sensor_energy'] = 0

            if id in self.linkStats.keys():
                self.res[id]['link_latency'] = self.linkStats[id]['latency']
                self.res[id]['link_energy'] = self.linkStats[id]['energy']
            else:
                self.res[id]['link_latency'] = 0
                self.res[id]['link_energy'] = 0

            if id in self.edgeStats.keys():
                self.res[id]['edge_latency'] = self.edgeStats[id]['latency']
                self.res[id]['edge_energy'] = self.edgeStats[id]['energy']
            else:
                self.res[id]['edge_latency'] = 0
                self.res[id]['edge_energy'] = 0

            self.res[id]['latency'] = self.res[id]['sensor_latency'] + self.res[id]['link_latency'] + self.res[id]['edge_latency']
            self.res[id]['energy'] = self.res[id]['sensor_energy'] + self.res[id]['link_energy'] + self.res[id]['edge_energy']

        # remove non-beneficial partitioning points based on bandwidth constraint
        filtered_pp = [layer.get_layer_name(False, True) for layer in self.dnn.partpoints_filtered]
        self.pp_res = {key:self.res[key] for key in self.res if key in filtered_pp}

    def export_csv(self, name) -> None:
        self._write_csv_files(name, self.res, self.pp_res)

    def _write_csv_files(self, runname : str ='test',
                        all_pp : Optional[dict] = None,
                        filtered_pp : Optional[dict] = None
        ) -> None:
        self._write_csv_file((runname + '_all.csv'), all_pp)
        self._write_csv_file((runname + '.csv'), filtered_pp)

    def _write_csv_file(self, filename : str, data : dict) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            header = [
                "No.",
                "Layer",
                "Output Size",
                "Latency [ms]",
                "Energy [mJ]",
                "Sensor Latency",
                "Sensor Energy",
                "Link Latency",
                "Link Energy",
                "Edge Latency",
                "Edge Energy"
            ]
            writer.writerow(header)
            for i, layer in enumerate(data.keys()):
                row = [
                    (i + 1),
                    layer,
                    str(data[layer]['output_size']),
                    str(data[layer]['latency']),
                    str(data[layer]['energy']),
                    str(data[layer]['sensor_latency']),
                    str(data[layer]['sensor_energy']),
                    str(data[layer]['link_latency']),
                    str(data[layer]['link_energy']),
                    str(data[layer]['edge_latency']),
                    str(data[layer]['edge_energy'])
                ]
                writer.writerow(row)
