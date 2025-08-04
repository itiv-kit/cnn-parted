from dataclasses import dataclass

from framework.stages.stage_base import Stage
from framework.stages.artifacts import Artifacts
from framework.helpers.config_helper import ConfigHelper
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

class SystemParser(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)
        conf_helper = ConfigHelper(self.config)
        node_components, link_components = conf_helper.get_system_components()
        self._update_artifacts(artifacts, node_components, link_components)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config

    def _update_artifacts(self, artifacts: Artifacts, nodes, links):
        artifacts.system_parser.nodes = nodes
        artifacts.system_parser.links = links

    
@dataclass
class SystemParserArtifacts:
    nodes: list[dict] = None
    links: list[dict] = None