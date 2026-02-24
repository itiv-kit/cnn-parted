from dataclasses import dataclass
from loguru import logger

from framework.link.link import Link
from framework.noc import NocTopology, FullyConnectedNoc, MeshNoc
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
        match self.config.system.noc.topology:
            case "fully_connected":
                noc_topo = FullyConnectedNoc(self.config, Link(link_components[0]))
            case "mesh":
                noc_topo = MeshNoc(self.config, Link(link_components[0]))
            case _:
                raise RuntimeError(f"Could not parse NoC topology, found {self.config.system.noc.topology}")
        self._update_artifacts(artifacts, node_components, link_components, noc_topo)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config

    def _update_artifacts(self, artifacts: Artifacts, nodes: list[NodeConfig], links: list[LinkConfig], noc_topo: NocTopology):
        artifacts.system_parser.nodes = nodes
        artifacts.system_parser.links = links
        artifacts.system_parser.noc = noc_topo

    
@dataclass
class SystemParserArtifacts:
    nodes: list[NodeConfig] = None
    links: list[LinkConfig] = None
    noc: NocTopology = None