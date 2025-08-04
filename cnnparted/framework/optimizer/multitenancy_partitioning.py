import os
import numpy as np
from numpy.random import default_rng
import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from joblib import Parallel, delayed

from framework.optimizer.optimizer import Optimizer
from framework.optimizer.partitioning_problem import PartitioningProblem
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult


# What this should do:
#   - given be a random, initial partitioning of the networks
#   - try to optimize EDP by moving partitions betwenn accelerators

class MultiTenancyPartitioning(Optimizer):
    def __init__(self, result: SystemResult):
        self.network_graphs = None
        self.system_result = result

    def _optimize(self):
        ...

    def _swap_random_layer(self):
        ...
    
    def _move_random_layer(self):
        ...
    
    def _eval_design_layer_set(self):
        ...
    
    def _gen_initial_binding(self):
        ...