from framework.constants import DNN_DICT
import yaml
from collections import defaultdict

class ConfigHelper:
    def __init__(self, fname : str):
        self.fname = fname
        self.config = self.__load_config()
    
    def __load_config(self) -> dict:
        with open(self.fname) as f:
            return yaml.load(f, Loader = yaml.SafeLoader)

    def get_config(self) -> dict:
        return self.config
    
    def get_system_components(self):
        components = self.config.get('components', [])
        sorted_components = sorted(components, key=lambda x: x.get('id', -1))
        
        node_components = []
        link_components = []
        for component in sorted_components:
            if 'timeloop' in component or 'device' in component:
                node_components.append(component)
            else:
                link_components.append(component)

        return node_components,link_components
    
    def print_all_keys(self,data_dict, indent=''):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                self.print_all_keys(value, indent + '  ')
            else:
                print(f"{indent}{key}")

    def get_constraints(self):
        try:
            max_size = self.config['constraints']['max_out_size']
        except KeyError:
            max_size = 0
        try:
            max_memory_size = self.config['constraints']['max_memory_size']
        except KeyError:
            max_memory_size = 0
        
        try:
            word_width = self.config['constraints']['word_width']
        except KeyError:
            word_width = 16 # default 2 bytes
        
        constraints = {
            "max_out_size": max_size,
            "max_memory_size": max_memory_size,
            "word_width": word_width
            }
        return constraints

    def get_node_memory_sizes(self, node_components ):
        memories=[]
        for node in node_components:
            memories.append( node["max_memory_size"])
        
        return memories

    def get_num_bytes(self):
        constraints = self.get_constraints()
        num_bytes = int(constraints["word_width"] / 8)  
        if constraints["word_width"] % 8 > 0:
            num_bytes += 1
        return num_bytes

    def get_model(self,main):
        try:
            model = DNN_DICT[self.config['neural-network']['name']]()
        except KeyError:
            print()
            print('\033[1m' + 'DNN not available - please use on of the supported networks:' + '\033[0m')
            for nn in [k for k in DNN_DICT.keys() if type(DNN_DICT[k]) == type(main)]: # only print functions
                print(nn)
            print()
            quit(1)

        return model
    
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
