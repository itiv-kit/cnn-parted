import threading
import time
import os

from .DNNAnalyzer import DNNAnalyzer
from framework.constants import ROOT_DIR

class ModuleThreadInterface(threading.Thread):
    def __init__(self, id : str, dnn : DNNAnalyzer, config : dict,reverse : bool, runname : str, show_progress : bool) -> None:
        threading.Thread.__init__(self)
        self.id = id
        self.dnn = dnn
        self.config = config
        self.runname = runname
        self.show_progress = show_progress
        self.reverse= reverse
        self.work_path= os.path.join(ROOT_DIR,self.runname)
        if not os.path.exists(self.work_path):
            os.makedirs(self.work_path)


        self.stats = {}

    def run(self) -> None:
        t0 = time.time()
        self._eval()
        t1 = time.time()
        self.stats['sim_time'] = t1 - t0

    def _eval(self) -> None:
        raise NotImplementedError

    def getStats(self) -> dict:
        return self.id ,self.stats
