import csv
import statistics
from collections import OrderedDict
from typing import Optional
import yaml
from .DNNAnalyzer import DNNAnalyzer

class Dif_Evaluator():
    def __init__(self, dnn : DNNAnalyzer, sensorStats : dict) -> None:
        self.dnn = dnn
        self.sensorStats = self._calc_stats(sensorStats)
        self._evaluate()

    def _calc_stats(self, stat : dict) -> None:
        avg_stats = {}
        for j in stat[0].keys():
            if isinstance(stat[0][j], dict):
                avg_stats[j] = {}
                for k in stat[0][j].keys():
                    if isinstance(stat[0][j][k], dict):
                        avg_stats[j][k] = {}
                        for m in stat[0][j][k].keys():
                            mdn = statistics.median([stat[n][j][k][m] for n in stat.keys()])
                            if len(stat) > 1:
                                std = statistics.stdev([stat[n][j][k][m] for n in stat.keys()])
                            else:
                                std = 0

                            avg_stats[j][k][m] = [mdn, std]
                    else:
                        mdn = statistics.median([stat[n][j][k] for n in stat.keys()])
                        if len(stat) > 1:
                            std = statistics.stdev([stat[n][j][k] for n in stat.keys()])
                        else:
                            std = 0

                        avg_stats[j][k] = [mdn, std]
            else:
                mdn = statistics.median([stat[n][j] for n in stat.keys()])
                if len(stat) > 1:
                    std = statistics.stdev([stat[n][j] for n in stat.keys()])
                else:
                    std = 0

                avg_stats[j] = [mdn, std]

        return avg_stats

    def _evaluate(self) -> None:
        self.res = OrderedDict()

        for layer in self.dnn.partition_points:
            id = layer.get_layer_name(False, True)

            self.res[id] = {}
            self.res[id]['output_size'] = layer.output_size

            if id in self.sensorStats.keys():
                self.res[id]['sensor_latency'] = self.sensorStats[id]['latency'][0]
                self.res[id]['sensor_latency_iqr'] = self.sensorStats[id]['latency_iqr'][0]
                self.res[id]['sensor_energy'] = self.sensorStats[id]['energy'][0]
            else:
                self.res[id]['sensor_latency'] = 0
                self.res[id]['sensor_latency_iqr'] = 0
                self.res[id]['sensor_energy'] = 0

        # remove non-beneficial partitioning points based on bandwidth constraint
        filtered_pp = [layer.get_layer_name(False, True) for layer in self.dnn.partpoints_filtered]
        self.pp_res = {key:self.res[key] for key in self.res if key in filtered_pp}


    def save_stats(self, run: str ,file_name:str)-> None   :
        self._save_stats(run,file_name, self.res)       

    def _save_stats(self,run : str,file_name : str, data : dict) -> None:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow("RunName: "+str(run))
            writer.writerow(["Sensor Node: " + str(self.sensorStats['sim_time'][0])])
            header = [
                "No.",
                "Layer",
                "Sensor Latency",
                "Sensor Energy",
            ]
            writer.writerow(header)
            for i, layer in enumerate(data.keys()):
                row = [
                    (i + 1),
                    layer,
                    str(data[layer]['sensor_latency']),
                    str(data[layer]['sensor_energy']),
                ]
                writer.writerow(row)


