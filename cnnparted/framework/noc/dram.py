from framework.helpers.dtypes import ComputeTask, TransferTask, NodeId, NocCoordinate 

class DramDummy:
    def __init__(self):
        # Nodes that are one hop away are considered neighbors
        self.neighbors: list[NocCoordinate] = []
        self.schedule: list[TransferTask] = []
        self.cur_time: float = 0.0
