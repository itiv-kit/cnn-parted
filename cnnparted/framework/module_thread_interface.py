import threading
import time

from framework.graph_analyzer import GraphAnalyzer
from framework.dse.interfaces.architecture_config import ArchitectureConfig
from framework.node.node_evaluator import NodeResult

class ModuleThreadInterface(threading.Thread):
    def __init__(self, id : str, ga : GraphAnalyzer, node_config : dict, 
                 work_dir: str, runname : str, progress : bool, 
                 acc_adaptor = None, save_results = True,
                 dse_system_config = {}) -> None:
        threading.Thread.__init__(self)
        self.id = id
        self.ga = ga
        self.dse_system_config = dse_system_config
        self.config = node_config
        self.work_dir = work_dir
        self.runname = runname
        self.progress = progress
        self.acc_adaptor = acc_adaptor
        self.save_results = save_results

        self.stats: NodeResult = None

    def run(self) -> None:
        t0 = time.time()
        self.eval_node()
        t1 = time.time()
        self.stats.sim_time = t1 - t0

    def _eval(self) -> None:
        raise NotImplementedError
    
    def eval_node(self):
        raise NotImplementedError

    def getStats(self) -> tuple[str, NodeResult]:
        return self.id ,self.stats
