from framework.helpers.config_helper import ConfigHelper
from .noc import NocLink, NocNode, NocTopologies

class MeshNoc(NocTopologies):
    ...
    def __init__(self, cfg: ConfigHelper):
        ...