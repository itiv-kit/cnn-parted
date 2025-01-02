import os
import numpy as np
import pandas as pd

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.stages.optimization.partitioning_optimization import PartitioningOptimization
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage("PartitioningOptimization", "GraphAnalysis")
class ExportPartitionResults(Stage):
    def __init__(self):
        super().__init__()

    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)
        self._export_files()
        self._write_log_file()
        self._update_artifacts(artifacts)

    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config
        self.work_dir = self.config["work_dir"]
        self.run_name = artifacts.args["run_name"]
        #self.n_constr = artifacts.get_stage_result(PartitioningOptimization, "n_constr")
        #self.n_var = artifacts.get_stage_result(PartitioningOptimization, "n_var")
        self.optimizer_cfg: PartitioningOptConfig = artifacts.get_stage_result(PartitioningOptimization, "optimizer_cfg")
        self.n_constr = self.optimizer_cfg.n_constr
        self.n_var = self.optimizer_cfg.n_var_part
        self.runtime = artifacts.step_runtime

        sol = artifacts.get_stage_result(PartitioningOptimization, "sol")
        if not "accuracy" in self.config:
            for i, p in enumerate(sol["nondom"]): # achieving aligned csv file
                sol["nondom"][i] = np.append(p, float(0))
            for i, p in enumerate(sol["dom"]): # achieving aligned csv file
                sol["dom"][i] = np.append(p, float(0))
        self.sol = sol

        self.schedules = artifacts.get_stage_result(GraphAnalysis, "ga").schedules
        self.num_platforms = self.config["num_platforms"]

    def _update_artifacts(self, artifacts: Artifacts):
        ...


    def _export_files(self) -> None:
        res = []
        objectives = []
        n_var = self.n_var
        n_constr = self.n_constr
        idx_max = 1 + self.num_platforms + 1 #schedule ID + design IDs + num_pp
        for pareto, sched in self.sol.items():
            for sd in sched:
                data = np.append(sd, pareto)
                data = data.astype('U256')
                data[0:idx_max] = data[0:idx_max].astype(float).astype(int)
                data[n_constr+1:n_constr+n_var+1] = data[n_constr+1:n_constr+n_var+1].astype(float).astype(int)
                for i in range(n_constr+1,int(n_var/2)+n_constr+1):
                    data[i] = self.schedules[int(data[0])][int(data[i])-1]
                res.append(data)
                objectives.append(data[n_constr+n_var+1:])
            if pareto == "nondom":
                self._write_files(res, objectives, "nondom")
        self._write_files(res, objectives, "all")

    def _write_files(self, result : list, objectives : list, name : str) -> None:
        hdr_front = ["Schedule_ID", "PPs"]
        # generate number of platforms, latency of platforms, energy of platforms, num of pps, num of platforms
        hdr_mid = []
        for i in range(self.num_platforms):
            hdr_mid.append("Design_ID" + str(i))
        for i in range(int(self.n_var / 2) + 1):
            hdr_mid.append("Latency_" + str(i))
        for i in range(int(self.n_var / 2) + 1):
            hdr_mid.append("Energy_" + str(i))
        for i in range(int(self.n_var / 2) + 1):
            hdr_mid.append("Link_Latency_" + str(i))
        for i in range(int(self.n_var / 2) + 1):
            hdr_mid.append("Link_Energy_" + str(i))
        for i in range(int(self.n_var / 2)):
            hdr_mid.append("PartitioningPoint_" + str(i))
        for i in range(int(self.n_var / 2) + 1):
            hdr_mid.append("Platform_" + str(i))
        if len(objectives[0]) == 7:
            hdr_back = ["Latency", "Energy", "Throughput", "Area", "Link_Latency", "Link_Energy", "Type"]
        else:
            hdr_back = ["Latency", "Energy", "Throughput", "Area", "Link_Latency", "Link_Energy", "Accuracy", "Type"]

        hdr = hdr_front + hdr_mid + hdr_back
        df = pd.DataFrame(result)
        df.to_csv( os.path.join(self.work_dir, self.run_name + "_result_" + name + ".csv"), header=hdr, index_label='idx')

        if len(objectives[0]) == 7:
            hdr = ["Latency", "Energy", "Throughput", "Area", "Link_Latency", "Link_Energy", "Type"]
        else:
            hdr = ["Latency", "Energy", "Throughput", "Area", "Link_Latency", "Link_Energy", "Accuracy", "Type"]
        df = pd.DataFrame(objectives)
        df.to_csv( os.path.join(self.work_dir, self.run_name + "_objectives_" + name + ".csv"), header=hdr, index_label='idx')

    def _write_log_file(self) -> int:
        log_file = os.path.join(self.work_dir, self.run_name + ".log")
        f = open(log_file, "a")

        # Runtime of each step
        step_runtimes = self.runtime
        step_runtimes = [x - step_runtimes[0] for x in step_runtimes[1:]]
        t_prev = 0
        for i, x in enumerate(step_runtimes, 1):
            f.write("Step " + str(i) +  ": " + str(x - t_prev) + ' s \n')
            t_prev = x

        # Number of solutions found
        num_pp_schemes = 0
        for pareto, scheme in self.sol.items():
            f.write(str(pareto) + ": " + str(len(scheme)) + '\n')
            num_pp_schemes += len(scheme)
        if num_pp_schemes > 0:
            num_real_pp = [int(scheme[self.num_platforms+1]) for scheme in self.sol["nondom"]]
            for i in range(0, max(num_real_pp)+1):
                f.write(str(i) + " Partitioning Point(s): " + str(num_real_pp.count(i)) + '\n')
        else:
            f.write("No valid partitioning found! \n")

        f.close()
        return num_pp_schemes
