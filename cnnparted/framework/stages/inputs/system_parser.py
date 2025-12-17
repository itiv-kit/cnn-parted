from dataclasses import dataclass
from loguru import logger

from framework.stages.stage_base import Stage
from framework.stages.artifacts import Artifacts
from framework.helpers.config import NodeConfig, LinkConfig

class SystemParser(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        logger.info("Parsing system configuration")
        self._take_artifacts(artifacts)
        node_components, link_components = self.config.get_system_components()
        self._update_artifacts(artifacts, node_components, link_components)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config

    def _update_artifacts(self, artifacts: Artifacts, nodes: list[NodeConfig], links: list[LinkConfig]):
        artifacts.system_parser.nodes = nodes
        artifacts.system_parser.links = links

    
@dataclass
class SystemParserArtifacts:
    nodes: list[NodeConfig] = None
    links: list[LinkConfig] = None