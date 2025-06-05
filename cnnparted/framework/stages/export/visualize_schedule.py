import os
import glob
import random
from itertools import cycle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import yaml

from framework.helpers.config_helper import ConfigHelper
from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.stages.optimization.partitioning_optimization import PartitioningOptimization
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage(PartitioningOptimization, GraphAnalysis)
class VisualizeSchedule(Stage):
    def __init__(self):
        super().__init__()


    def _take_artifacts(self, artifacts):
        self.config = artifacts.config
        self.work_dir = self.config["work_dir"]
        self.run_name = artifacts.args["run_name"]
        self.n_constr = artifacts.get_stage_result(PartitioningOptimization, "n_constr")
        self.n_var = artifacts.get_stage_result(PartitioningOptimization, "n_var")

        sol = artifacts.get_stage_result(PartitioningOptimization, "sol")
        if not "accuracy" in self.config:
            for i, p in enumerate(sol["nondom"]): # achieving aligned csv file
                sol["nondom"][i] = np.append(p, float(0))
            for i, p in enumerate(sol["dom"]): # achieving aligned csv file
                sol["dom"][i] = np.append(p, float(0))
        self.sol = sol
        self.schedules = artifacts.get_stage_result(GraphAnalysis, "ga").schedules
        self.num_platforms = self.config["num_platforms"]
        self.num_pp = artifacts.config["num_pp"]
        self.n_var_part = (self.num_pp + 1) + (self.num_pp) # platform IDs + mapping partition to platform
        if self.num_pp == 0:
            self.n_var_part_metrics = 1
            self.n_var_part_points = 1
        else:
            self.n_var_part_metrics = self.num_pp + 1
            self.n_var_part_points = self.num_pp
        self.n_var_dse  = self.num_platforms # design IDs

        self.res = []
        self.objectives = []
        self.nondom_res = []
        self.nondom_objectives = []
        #n_var_part = self.n_var - self.num_platforms
        #self.n_var_part = n_var_part
        n_constr = self.n_constr
        hdr_front = 1 + 1
        for pareto, sched in self.sol.items():
            for sd in sched:
                data = np.append(sd, pareto)
                data = data.astype('U256')
                data[0:hdr_front] = data[0:hdr_front].astype(float).astype(int)
                data[n_constr+1:n_constr+self.n_var_part+1] = data[n_constr+1:n_constr+self.n_var_part+1].astype(float).astype(int)
                for i in range(n_constr+1,int(self.n_var_part/2)+n_constr+1):
                    data[i] = self.schedules[int(data[0])][int(data[i])-1]
                self.res.append(data)
                self.objectives.append(data[n_constr+self.n_var+1:])
            if pareto == "nondom":
                self.nondom_res.append(data)
                self.nondom_objectives.append(data[n_constr+self.n_var+1:])

        self.hdr_front_len = 2 + self.num_platforms
        self.idx_part_latency = np.s_[self.hdr_front_len : self.hdr_front_len+self.n_var_part_metrics]
        self.idx_part_energy = np.s_[self.hdr_front_len+self.n_var_part_metrics : self.hdr_front_len+self.n_var_part_metrics*2]
        self.idx_link_latency = np.s_[self.hdr_front_len+self.n_var_part_metrics*2 : self.hdr_front_len+self.n_var_part_metrics*3]
        self.idx_link_energy = np.s_[self.hdr_front_len+self.n_var_part_metrics*3 : self.hdr_front_len+self.n_var_part_metrics*4]
        self.idx_part_points = np.s_[self.hdr_front_len+self.n_var_part_metrics*4 : self.hdr_front_len+self.n_var_part_metrics*4+self.n_var_part_points]
        self.idx_part_mapping= np.s_[self.hdr_front_len+self.n_var_part_metrics*4+self.n_var_part_points : self.hdr_front_len+self.n_var_part_metrics*5+self.n_var_part_points]
        #self.idx_design_ids = np.s_[1+self.n_constr+self.n_var_part:1+self.n_constr+self.n_var_part+self.num_platforms] # n_constr already includeds num_pp, os we dont add hdr_front_len
        self.idx_design_ids = np.s_[2:2+self.num_platforms]

    def run(self, artifacts):
        self._take_artifacts(artifacts)
        solution = random.choice(self.res)
        part_latency = solution[self.idx_part_latency]
        part_points = solution[self.idx_part_points]
        part_points = np.append(part_points, 'output') #add last layer to make iteration easier
        part_mapping = solution[self.idx_part_mapping]
        design_ids = solution[self.idx_design_ids]

        eval_dir = os.path.join(self.work_dir, "system_evaluation")

        cur_time = 0
        sys, _ = ConfigHelper(self.config).get_system_components()
        platform_names = []
        for node in sys:
            name = node["evaluation"]["accelerator"]
            platform_names.append(name)
        
        fig = go.Figure()
        for lat, pp, mapping in zip(part_latency, part_points, part_mapping):
            lat = float(lat)
            mapping = int(mapping)
            design_id = design_ids[mapping].astype(float).astype(int)
            design_info = design_id

            # Check if there is more specific design info available
            platform_dir = glob.glob(os.path.join(eval_dir, str(mapping)+"*"))
            if platform_dir:
                design_dir = os.path.join(platform_dir[0], f"design{design_id}")
                if os.path.isdir(design_dir):
                    with open(os.path.join(design_dir, "arch_config.yaml"), "r") as f:
                        design_info = yaml.safe_load(f)
            
            hovertext = (
                f"<b>Latency:</b> {lat:.2e}<br>"
                f"<b>Start:</b> {cur_time:.2e}<br>"
                f"<b>End:</b> {cur_time+lat:.2e}<br>"
                f"<b>Design Id:</b> {design_info}<br>"
            )
            bar = go.Bar(
                base=[cur_time],
                x=[lat],
                y=[mapping],
                orientation="h",
                width=0.25,
                hovertext=hovertext,
            )
            cur_time += lat
            fig.add_trace(bar)
    
        fig.update_layout(
            title_text=( f"Execution Schedule"),
            barmode="stack",
            showlegend=False,
        )
        fig.update_yaxes(
            title="Platform [-]",
            tickvals=list(range(self.num_platforms)),
            ticktext=platform_names,
        )
        fig.update_xaxes(
            title="Time [ms]",
        )
        
        fig_dir = os.path.join(self.work_dir, "figures", "schedule")
        fig_file_html = os.path.join(fig_dir, "schedule.html")
        fig_file_json = os.path.join(fig_dir, "schedule.json")
        fig_file_png = os.path.join(fig_dir, "schedule.png")
        os.makedirs(fig_dir, exist_ok=True)
        fig.write_html(fig_file_html)
        fig.write_image(fig_file_png)
        fig.write_json(fig_file_json)


    def _update_artifacts(self, artifacts):
        pass
