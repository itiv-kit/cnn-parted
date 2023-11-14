import threading
from framework.ModuleThreadInterface import ModuleThreadInterface
from .LinkThread import LinkThread
from .MemoryNodeThread import MemoryNodeThread
class LinkComputationThread(ModuleThreadInterface):
    def __init__(self, id, dnn, config, reverse, runname, show_progress):
        super().__init__(id, dnn, config, reverse, runname, show_progress)  # Initialize the superclass
        self._select_strategy()

    def _select_strategy(self):
        # Select the appropriate strategy
        if 'memory' in self.config:
            self.strategy = MemoryNodeThread(self.id, self.dnn,self.config,self.reverse,self.runname,self.show_progress)
        elif 'ethernet' in self.config or 'noi' in self.config:
            self.strategy = LinkThread(self.id, self.dnn,self.config,self.reverse,self.runname,self.show_progress)
        else:
            raise ValueError("Invalid configuration, cannot select strategy")

    # def run(self):
    #     try:
    #         self._select_strategy()
    #         self.strategy._eval()  # Run the strategy (assumes _eval() is the computational workhorse)
    #         self.stats = self.strategy.stats
    #     except Exception as e:
    #         print(f"Error during computation: {e}")

    # def getStats(self):
    #     # Return the computed stats
    #     return self.stats

    def _eval(self) -> None:
        self.strategy.eval()
        self.stats = self.strategy.stats
