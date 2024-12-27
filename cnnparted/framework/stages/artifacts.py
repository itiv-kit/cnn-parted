import importlib

from framework.stages.stage_base import Stage


class Artifacts:
    def __init__(self, config: dict, args: dict, device: str, step_runtime: list):
        self.config = config
        self.args = args
        self.device = device
        self.step_runtime = step_runtime

        self.parse_stages()
    
    def parse_stages(self):
        package = importlib.import_module("framework.stages")
        name_to_class_map = {stage.__name__: stage for stage in package.__all__}

        self.stage_names = self.config["pipeline"]["stages"]
        self.stages: dict[Stage, StageArtifacts] = {}
        for stage in self.stage_names:
            artifact = StageArtifacts()
            artifact.name = stage
            self.stages[name_to_class_map[stage]] = artifact

    def get_stage_result(self, stage: Stage, name: str):
        return self.stages[stage].get_result(name)

    # Get all results of a stage 
    def get_stage_results(self, stage: Stage):
        return self.stages[stage]

    def set_stage_result(self, stage: Stage, name: str, value):
        self.stages[stage].set_result(name, value)



class StageArtifacts:
    def __init__(self) -> None:
        self.name = None

    def get_result(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            k = vars(self).keys()
            raise AttributeError(f"Field {name} not found in artifacts for \"{self.name}\". Fields are: {k}")

    def set_result(self, name: str, value):
        setattr(self, name, value)