from framework.graph_analyzer import GraphAnalyzer
from framework.optimizer.optimizer import Optimizer

class DesignPartitioningOptimizer(Optimizer):

    def __init__(self, ga: GraphAnalyzer, 
                 num_pp: int, link_components: list, progress: bool,
                 desing_optimizer: Optimizer, partitioning_optimizer: Optimizer):
        self.design_optimizer = desing_optimizer
        self.part_optimizer = partitioning_optimizer
        self.work_dir = ga.work_dir
        self.run_name = ga.run_name
        self.schedules = ga.schedules
        self.num_pp = num_pp
        self.link_confs = link_components
        self.progress = progress
        nodes = len(ga.schedules[0])

    def optimize(self, q_constr, conf):
        
        self.design_lut = self.design_optimizer.optimize()

        self.system_results = self.part_optimizer.optimize(q_constr, conf)

    def _optimize_single(self):
        ...


        