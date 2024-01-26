import threading
import time

from framework import GraphAnalyzer

class ModuleThreadInterface(threading.Thread):
    def __init__(self, id : str, ga : GraphAnalyzer, config : dict, runname : str, progress : bool) -> None:
        threading.Thread.__init__(self)
        self.id = id
        self.ga = ga
        self.config = config
        self.runname = runname
        self.progress = progress

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
