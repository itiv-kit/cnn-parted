from ast import Lambda
import yaml

class ConfigHelper:
    def __init__(self, conf_in : str | dict):
        if isinstance(conf_in, str):
            self.fname = conf_in
            self.config = self.__load_config()
        elif isinstance(conf_in, dict):
            self.fname = ""
            self.config = conf_in
        else:
            raise RuntimeError(f"Invalid argument to ConfigHelper. Expected str or dict but received {type(conf_in)}")
        

    def __load_config(self) -> dict:
        with open(self.fname) as f:
            return yaml.load(f, Loader = yaml.SafeLoader)

    def get_config(self) -> dict:
        return self.config

    def get_system_components(self):
        if "components" in self.config.keys():
            # Old config style: determine compute vs link by content of the entry
            components = self.config.get('components', [])
            sorted_components = sorted(components, key=lambda x: x.get('id', -1))

            node_components = []
            link_components = []
            for component in sorted_components:
                if 'timeloop' in component or 'mnsim' in component or 'device' in component or 'zigzag' in component:
                    node_components.append(component)
                else:
                    link_components.append(component)
        elif system := self.config.get("system"):
            # New config style: dedicated fields for compute and link nodes
            nodes = system.get("compute", [])
            links = system.get("link", [])
            node_components = sorted(nodes, key=lambda x: x.get("id", -1))
            link_components = sorted(links, key=lambda x: x.get("id", -1))
        else:
            # No system config found
            node_components = []
            link_components = []

        return node_components,link_components

    def print_all_keys(self,data_dict, indent=''):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                self.print_all_keys(value, indent + '  ')
            else:
                print(f"{indent}{key}")

    def get_node_memory_sizes(self, node_components ):
        memories=[]
        for node in node_components:
            memories.append( node["max_memory_size"])

        return memories

    def get_optimization_objectives(self, nodes, links):
        try:
            id = self.config['optimization-objectives']['device']
        except KeyError:
            id = '0'

        try:
            metric = self.config['optimization-objectives']['metric']
        except KeyError:
            metric = 'energy'

        if any(str(node.get('id')) == id for node in nodes):
            name = "Node-" + str(id)

        elif any(str(link.get('id')) == id for link in links):
            name = "Link-" + str(id)
        else:
            raise ValueError(f"ID {id} not found in nodes or links.")

        key_name = f"{name}_{metric}"


        return key_name
