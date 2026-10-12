from framework.node.node_evaluator import NodeEvaluator

class GenericNode(NodeEvaluator):
    def __init__(self, config):
        self.config = config

    def set_workdir(self, work_dir: str, runname: str, id: int):
        return super().set_workdir(work_dir, runname, id)

    def run(self, layers: list):
        raise NotImplementedError