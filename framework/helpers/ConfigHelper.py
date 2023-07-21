from framework.constants import DNN_DICT
import yaml

class ConfigHelper:
    def __init__(self, fname : str):
        self.fname = fname
        self.config = self.__load_config()
    
    def __load_config(self) -> dict:
        with open(self.fname) as f:
            return yaml.load(f, Loader = yaml.SafeLoader)

    def get_config(self) -> dict:
        return self.config

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
    
    def get_optimization_objectives(self):
        try:
            device = self.config['optimization_objectives']['device']
        except KeyError:
            device = 'sensor'
        try:
            metric =self.config['optimization_objectives']['metric']
        except KeyError:
            metric = 'energy'
        
        optimization_objectives = {
            "device": device,
            "metric": metric
            }
        return optimization_objectives