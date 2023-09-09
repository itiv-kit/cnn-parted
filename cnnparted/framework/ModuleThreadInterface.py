import threading
import time

from .DNNAnalyzer import DNNAnalyzer

class ModuleThreadInterface(threading.Thread):
    def __init__(self, name : str, dnn : DNNAnalyzer, config : dict,reverse : bool, runname : str, show_progress : bool) -> None:
        threading.Thread.__init__(self)
        self.name = name
        self.dnn = dnn
        self.config = config
        self.runname = runname
        self.show_progress = show_progress
        self.reverse= reverse

        self.stats = {}

    def run(self) -> None:
        t0 = time.time()
        self._eval()
        t1 = time.time()
        self.stats['sim_time'] = t1 - t0

    def _eval(self) -> None:
        raise NotImplementedError

    def getStats(self) -> dict:
        return self.stats
