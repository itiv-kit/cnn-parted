import csv
import statistics
from collections import OrderedDict
from typing import Optional
import numpy as np
from typing import List
from .DNNAnalyzer import DNNAnalyzer


class Evaluator:
    def __init__(
        self,
        dnn: DNNAnalyzer,
        nodeStats: List[dict],
        linkStats: List[dict],
        memoryStats:dict,
        accStats: dict,
    ) -> None:
        self.dnn = dnn
        self.nodeStats= nodeStats

        self.linkStats= linkStats
        self.memoryStats= memoryStats
        allStats = OrderedDict()
        allStats.update(nodeStats)
        allStats.update(linkStats)
        allStats.update(memoryStats)
        self.memory_name = "DDR3" 
        self.sorted_allStats = OrderedDict(sorted(allStats.items()))
        self.first_id = next(iter(self.sorted_allStats))
        self.Stats={}
        for id  , cmp in self.sorted_allStats.items():
            self.Stats[id]= self._calc_stats(cmp)

        # dont know yet how it is implemented, so it should be probably modified
        self.accStats = accStats

        self.input_size= dnn.input_size
        self.num_bytes = dnn.num_bytes
        self.part_point_memory= dnn.part_point_memory
        self._evaluate()


    def print_sim_time(self) -> None:
        print()
        print("Median Simulation Time")
        print("===========================================")
        print("DNN Analyzer: ", self.dnn.stats["sim_time"], "s")

        if "sim_time" in self.accStats.keys():
            print("Accuracy Eval:", self.accStats["sim_time"], "s")

        
        for id,cmp in self.Stats.items():
            if id in self.nodeStats:
                name = "Node-"+str(id)
            elif id in self.linkStats:
                name= "link-"+ str(id)
            else:
                name= self.memory_name 
            print(
            name,"   :",
            cmp["sim_time"][0],
            "s (stdev: ",
            cmp["sim_time"][1],
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
        self.stats_names=[]
        for layer in self.dnn.partition_points:
            layer_name = layer.get("name")
            

            if layer_name == self.dnn.max_part_point:
                last_layer_fits_memory_found = True
            elif last_layer_fits_memory_found:
                break

            self.res[layer_name] = {}
            self.res[layer_name]["output_size"] = layer.get("output_size")
            #self.res[layer_name]["sensor_memory"]= self.part_point_memory[layer_name]#we need more for more than 2 accelerator setting


            for id,cmp in self.Stats.items():#for id,cmp in self.sorted_allStats.items():

                if id in self.memoryStats:
                    mem_name = self.memory_name
                    if layer_name in cmp.keys():                              
                        self.res[layer_name][f"{mem_name}_write_latency"] = cmp[layer_name]["write_latency_ms"][0]
                        self.res[layer_name][f"{mem_name}_write_energy"] = cmp[layer_name]["write_energy"][0]
                        self.res[layer_name][f"{mem_name}_read_latency"] = cmp[layer_name]["read_latency_ms"][0]
                        self.res[layer_name][f"{mem_name}_read_energy"] = cmp[layer_name]["read_energy"][0]

                    else:
                        self.res[layer_name][f"{mem_name}_write_latency"]=0
                        self.res[layer_name][f"{mem_name}_wrtie_energy"] =0
                        self.res[layer_name][f"{mem_name}_read_latency"] =0
                        self.res[layer_name][f"{mem_name}_read_energy"]  =0
                else:
                    if id in self.nodeStats:
                        name = "Node-"+str(id)
                        if layer_name in cmp.keys():
                            self.res[layer_name][f"{name}_latency"] = cmp[layer_name]["latency"][0]
                            self.res[layer_name][f"{name}_latency_iqr"] = cmp[layer_name]["latency_iqr"][0]
                            self.res[layer_name][f"{name}_energy"] = cmp[layer_name]["energy"][0]
                            self.res[layer_name][f"{name}_memory"]= self.part_point_memory[layer_name]
                        else:
                            self.res[layer_name][f"{name}_latency"] = 0
                            self.res[layer_name][f"{name}_latency_iqr"] = 0
                            self.res[layer_name][f"{name}_energy"] = 0

                    else:
                        name= "Link-"+ str(id)
                        if layer_name in cmp.keys():            
                            self.res[layer_name][f"{name}_latency"] = cmp[layer_name]["latency"][0]
                            self.res[layer_name][f"{name}_energy"] = cmp[layer_name]["energy"][0]
                        else:
                            self.res[layer_name][f"{name}_latency"] = 0
                            self.res[layer_name][f"{name}_energy"] = 0

                    if name not in self.stats_names:
                        self.stats_names.append(name)            
        

            if layer_name in self.accStats.keys():
                self.res[layer_name]["accuracy"] = self.accStats[layer_name]
            else:
                self.res[layer_name]["accuracy"] = 0

            total_latency=0
            for name in self.stats_names:
                latency_key = f"{name}_latency"
                if latency_key in self.res[layer_name]:
                    total_latency += self.res[layer_name][latency_key]

            self.res[layer_name]["latency"] = total_latency

            total_energy=0
            for name in self.stats_names:
                energy_key = f"{name}_energy"
                if energy_key in self.res[layer_name]:
                    total_energy += self.res[layer_name][latency_key]

            self.res[layer_name]["energy"] = total_energy

            throughputs=[]      
            for id,cmp in self.Stats.items():#for id,cmp in self.sorted_allStats.items():
                if id in self.nodeStats:
                    name = "Node-"+str(id)
                else:
                    name = "Link-" + str(id)

                size = self.input_size if id==self.first_id else self.res[layer_name]["output_size"]
    
                throughput = np.prod(size) / float(self.res[layer_name][latency_key])
                throughputs.append(throughput) 
            
            min_throughput = min(throughputs)
            self.res[layer_name]["throughput"] = round(min_throughput * self.num_bytes / 1000, 2)  # MBps = 1000/1e6 B/ms

        filtered_pp = [layer.get("name") for layer in self.dnn.partpoints_filtered]
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
            base_header = ["No.", "Layer", "Output Size", "Accuracy", "Latency [ms]", "Energy [mJ]"]
            dynamic_header = []
        
            for name in self.stats_names:
                dynamic_header.extend([
                    f"{name} Latency",
                    f"{name} Latency IQR",
                    f"{name} Energy"
                ])
                if "Node" in name:  # Adding memory header for nodes
                    dynamic_header.append(f"{name} Memory[bytes]")

            dynamic_header.append("throughput[MBps]")
            memory_header=["DDR3 Write Latency[ms]","DDR3 Write Energy[mJ]", "DDR3 Read Latency[ms]","DDR3 Read Energy[mJ]"]

            header = base_header + dynamic_header + memory_header
            writer.writerow(header)

            for i, layer in enumerate(data.keys()):
                # Dynamic row generation
                base_row = [
                    i + 1,
                    layer,
                    str(data[layer]["output_size"]),
                    str(data[layer]["accuracy"]),
                    str(data[layer]["latency"]),
                    str(data[layer]["energy"])
                ]

                dynamic_row = []
                for name in self.stats_names:
                    dynamic_row.extend([
                        str(data[layer].get(f"{name}_latency", "")),
                        str(data[layer].get(f"{name}_latency_iqr", "")),
                        str(data[layer].get(f"{name}_energy", ""))
                    ])
                    if "Node" in name:  # Extracting memory data for nodes
                        dynamic_row.append(str(data[layer].get(f"{name}_memory", "")))

                dynamic_row.append(str(data[layer]["throughput"]))

                memory_row=[]
                name = self.memory_name
                memory_row.extend([
                    str(data[layer].get(f"{name}_write_latency", "")),
                    str(data[layer].get(f"{name}_write_energy", "")),
                    str(data[layer].get(f"{name}_read_latency", "")),
                    str(data[layer].get(f"{name}_read_energy", ""))

                ])

                row = base_row + dynamic_row + memory_row
                writer.writerow(row)

    def get_all_layer_stats(self) -> OrderedDict:
     output = OrderedDict()

     for i, layer in enumerate(self.pp_res.keys()):
         if i == 0:
             continue

         output[i] = {}
         output[i]["layer"] = layer
         output[i]["energy"] = self.pp_res[layer]["energy"]
         output[i]["latency"] = self.pp_res[layer]["latency"]
         output[i][f"{self.memory_name}_write_latency"] = self.pp_res[layer][f"{self.memory_name}_write_latency"]
         output[i][f"{self.memory_name}_write_energy"] = self.pp_res[layer][f"{self.memory_name}_write_energy"]
         output[i][f"{self.memory_name}_read_latency"] = self.pp_res[layer][f"{self.memory_name}_read_latency"]
         output[i][f"{self.memory_name}_read_energy"] = self.pp_res[layer][f"{self.memory_name}_read_energy"]

         for name in self.stats_names:
             latency_key = f"{name}_latency"
             energy_key = f"{name}_energy"
             memory_key = f"{name}_memory"  # This might not exist for every name

             output[i][latency_key] = self.pp_res[layer].get(latency_key, 0)
             output[i][energy_key] = self.pp_res[layer].get(energy_key, 0)

             if "Node" in name:
                 output[i][memory_key] = self.pp_res[layer].get(memory_key, 0)

         output[i]["throughput"] = self.pp_res[layer].get("throughput", 0)

     return output