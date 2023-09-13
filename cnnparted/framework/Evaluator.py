import csv
import statistics
from collections import OrderedDict
from typing import Optional
import numpy as np
from .DNNAnalyzer import DNNAnalyzer


class Evaluator:
    def __init__(
        self,
        dnn: DNNAnalyzer,
        sensorStats: dict,
        linkStats: dict,
        edgeStats: dict,
        accStats: dict,
    ) -> None:
        self.dnn = dnn
        self.sensorStats = self._calc_stats(sensorStats)
        self.linkStats = self._calc_stats(linkStats)
        self.edgeStats = self._calc_stats(edgeStats)
        self.accStats = accStats
        self.input_size= dnn.input_size
        self.part_point_memory= dnn.part_point_memory
        self._evaluate()

    def print_sim_time(self) -> None:
        print()
        print("Median Simulation Time")
        print("===========================================")
        print("DNN Analyzer: ", self.dnn.stats["sim_time"], "s")

        if "sim_time" in self.accStats.keys():
            print("Accuracy Eval:", self.accStats["sim_time"], "s")

        print(
            "Sensor Node:  ",
            self.sensorStats["sim_time"][0],
            "s (stdev: ",
            self.sensorStats["sim_time"][1],
            "s)",
        )
        print(
            "Link:         ",
            self.linkStats["sim_time"][0],
            "s (stdev: ",
            self.linkStats["sim_time"][1],
            "s)",
        )
        print(
            "Edge Node:    ",
            self.edgeStats["sim_time"][0],
            "s (stdev: ",
            self.edgeStats["sim_time"][1],
            "s)",
        )
        print()

    def _calc_stats(self, stat: dict) -> None:
        avg_stats = {}
        for j in stat[0].keys():
            if isinstance(stat[0][j], dict):
                avg_stats[j] = {}
                for k in stat[0][j].keys():
                    if isinstance(stat[0][j][k], dict):
                        avg_stats[j][k] = {}
                        for m in stat[0][j][k].keys():
                            mdn = statistics.median(
                                [stat[n][j][k][m] for n in stat.keys()]
                            )
                            if len(stat) > 1:
                                std = statistics.stdev(
                                    [stat[n][j][k][m] for n in stat.keys()]
                                )
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
        last_layer_fits_memory_found = False
        for layer in self.dnn.partition_points:
            id = layer.get("name")
            

            if id == self.dnn.max_part_point:
                last_layer_fits_memory_found = True
            elif last_layer_fits_memory_found:
                break

            self.res[id] = {}
            self.res[id]["output_size"] = layer.get("output_size")
            self.res[id]["sensor_memory"]= self.part_point_memory[id]
          
            if id in self.sensorStats.keys():
                self.res[id]["sensor_latency"] = self.sensorStats[id]["latency"][0]
                self.res[id]["sensor_latency_iqr"] = self.sensorStats[id][
                    "latency_iqr"
                ][0]
                self.res[id]["sensor_energy"] = self.sensorStats[id]["energy"][0]
            else:
                self.res[id]["sensor_latency"] = 0
                self.res[id]["sensor_latency_iqr"] = 0
                self.res[id]["sensor_energy"] = 0

            if id in self.linkStats.keys():
                self.res[id]["link_latency"] = self.linkStats[id]["latency"][0]
                self.res[id]["link_energy"] = self.linkStats[id]["energy"][0]
            else:
                self.res[id]["link_latency"] = 0
                self.res[id]["link_energy"] = 0

            if id in self.edgeStats.keys():
                self.res[id]["edge_latency"] = self.edgeStats[id]["latency"][0]
                self.res[id]["edge_latency_iqr"] = self.edgeStats[id]["latency_iqr"][0]
                self.res[id]["edge_energy"] = self.edgeStats[id]["energy"][0]
            else:
                self.res[id]["edge_latency"] = 0
                self.res[id]["edge_latency_iqr"] = 0
                self.res[id]["edge_energy"] = 0

            if id in self.accStats.keys():
                self.res[id]["accuracy"] = self.accStats[id]
            else:
                self.res[id]["accuracy"] = 0

            self.res[id]["latency"] = (
                self.res[id]["sensor_latency"]
                + self.res[id]["link_latency"]
                + self.res[id]["edge_latency"]
            )
            self.res[id]["energy"] = (
                self.res[id]["sensor_energy"]
                + self.res[id]["link_energy"]
                + self.res[id]["edge_energy"]
            )

            sensor_thrp = np.prod(self.input_size) / float(self.pp_res[id]["sensor_latency"])
            link_thrp= np.prod(self.pp_res[id]["output_size"]) / float(self.pp_res[id]["link_latency"])
            edge_thrp= np.prod(self.pp_res[id]["output_size"]) / float(self.pp_res[id]["edge_latency"])
            
            self.res[id]["throughput"] = min(sensor_thrp,link_thrp , edge_thrp)

        # remove non-beneficial partitioning points based on bandwidth constraint
        filtered_pp = [layer.get("name") for layer in self.dnn.partpoints_filtered]
        # print(filtered_pp)
        self.pp_res = {key: self.res[key] for key in self.res if key in filtered_pp}

    def export_csv(self, name) -> None:
        self._write_csv_files(name, self.res, self.pp_res)

    def _write_csv_files(
        self,
        runname: str = "test",
        all_pp: Optional[dict] = None,
        filtered_pp: Optional[dict] = None,
    ) -> None:
        self._write_csv_file((runname + "_all.csv"), all_pp)
        self._write_csv_file((runname + ".csv"), filtered_pp)

    def _write_csv_file(self, filename: str, data: dict) -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            header = [
                "No.",
                "Layer",
                "Output Size",
                "Accuracy",
                "Latency [ms]",
                "Energy [mJ]",
                "Sensor Latency",
                "Sensor Latency IQR",
                "Sensor Energy",
                "Sensor Memory[bytes]",
                "Link Latency",
                "Link Energy",
                "Edge Latency",
                "Edge Latency IQR",
                "Edge Energy",
                "throughput",
            ]
            writer.writerow(header)
            for i, layer in enumerate(data.keys()):
                row = [
                    (i + 1),
                    layer,
                    str(data[layer]["output_size"]),
                    str(data[layer]["accuracy"]),
                    str(data[layer]["latency"]),
                    str(data[layer]["energy"]),
                    str(data[layer]["sensor_latency"]),
                    str(data[layer]["sensor_latency_iqr"]),
                    str(data[layer]["sensor_energy"]),
                    str(data[layer]["sensor_memory"]),
                    str(data[layer]["link_latency"]),
                    str(data[layer]["link_energy"]),
                    str(data[layer]["edge_latency"]),
                    str(data[layer]["edge_latency_iqr"]),
                    str(data[layer]["edge_energy"]),
                    str(data[layer]["throughput"]),
                ]
                writer.writerow(row)

    def get_all_layer_stats(self) -> OrderedDict:
        output = OrderedDict()
        # print(self.pp_res.keys())

        for i, layer in enumerate(self.pp_res.keys()):
            if i == 0:
                continue

            output[i] = {}
            output[i]["layer"] = layer
            output[i]["energy"] = self.pp_res[layer]["energy"]
            output[i]["latency"] = self.pp_res[layer]["latency"]

            output[i]["sensor_latency"] = self.pp_res[layer]["sensor_latency"]
            output[i]["sensor_energy"] = self.pp_res[layer]["sensor_energy"]
            output[i]["sensor_memory"] = self.pp_res[layer]["sensor_memory"]


            output[i]["link_latency"] = self.pp_res[layer]["link_latency"]
            output[i]["link_energy"] = self.pp_res[layer]["link_energy"]

            output[i]["edge_latency"] = self.pp_res[layer]["edge_latency"]
            output[i]["edge_energy"] = self.pp_res[layer]["edge_energy"]   
            output[i]["throughput"] =  self.pp_res[layer]["throughput"]
        return output
