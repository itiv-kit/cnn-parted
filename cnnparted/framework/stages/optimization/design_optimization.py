from dataclasses import dataclass
from loguru import logger

from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import SystemResult
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.robustness_optimization import RobustnessOptimization
from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts

from framework.optimizer.design_optimizer import DesignOptimizer
from framework.optimizer.config.design_opt_config import DesignOptConfig
from framework.optimizer.partitioning_optimizer import PartitioningOptimizer
from framework.optimizer.design_problem import DesignProblem

from framework.helpers.config_helper import ConfigHelper

@register_required_stage(GraphAnalysis, SystemParser)
class DesignPartitioningOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        logger.info("Running design optimization stage")
        self._take_artifacts(artifacts)

        cfg_helper = ConfigHelper(self.config)
        node_components, link_components = cfg_helper.get_system_components()

        # If no exhaustive search should be performed, instantiate the
        # problem for evaluation
        design_problem = None
        design_opt_config = None #TODO
        if self.config.dse.optimizer.algorithm != "exhaustive": #["dse"]["optimizer"]["algorithm"] != "exhaustive":
            node_constraints = cfg_helper.get_node_constraints()
            node_mem_steps = cfg_helper.get_node_mem_steps()

            # TODO: Design Problem and Optimizer selected by config
            design_problem = DesignProblem(node_components, link_components, 
                                        node_constraints, node_mem_steps,
                                        self.q_constr,
                                        artifacts, PartitioningOptimizer,
                                        self.config.dse)
            design_opt_config = design_problem.design_opt_config

        design_optimizer = DesignOptimizer(node_components, 
                                           design_problem, 
                                           self.config.dse, 
                                           self.work_dir,
                                           self.ga,
                                           self.run_name,
                                           self.show_progress)

        node_stats = design_optimizer.optimize(self.q_constr, artifacts.config)

        top_k = self.config.dse.top_k
        metric = self.config.dse.optimization  #["dse"].get("optimization", "edap")
        strict_mode = self.config.dse.prune_strict
        for id, stats in node_stats.platforms.items():
            stats.prune_designs(top_k, metric, strict_mode)

        self._update_artifacts(artifacts, node_stats, design_opt_config)

    
    def _take_artifacts(self, artifacts: Artifacts):
        self.run_name = artifacts.config.args.run_name
        self.ga = artifacts.graph_analysis.ga
        self.show_progress = artifacts.config.args.show_progress
        self.num_pp = artifacts.config.general.num_pp
        self.link_components = artifacts.system_parser.links
        self.work_dir = artifacts.config.general.work_dir

        self.config = artifacts.config

        if RobustnessOptimization in artifacts.stages:
            self.q_constr = artifacts.robustness_optimization.q_constr
        else:
            self.q_constr = {}
        

    def _update_artifacts(self, artifacts: Artifacts, node_stats: SystemResult, optimizer_config):
        artifacts.config.system.num_platforms = node_stats.get_num_platforms()
        artifacts.design_optimization.node_stats = node_stats
        artifacts.design_optimization.optimizer_cfg = optimizer_config

        #num_pp = artifacts.config["general"]["num_pp"]
        #if num_pp == -1:
            #platform_id = node_stats.get_platform_ids()[0]
            #design = node_stats[platform_id].designs[0]
            #network =  node_stats[platform_id].designs[0].get_networks()[0] #TODO Multiple networks
            #num_pp = len(design.networks[network].layers) - 1
        #elif len(node_stats) == 1:
        #    num_pp = 0
        #artifacts.config["num_pp"] = num_pp

@dataclass
class DesignOptimizationArtifacts:
    node_stats: SystemResult = None
    optimizer_cfg: DesignOptConfig = None