import pathlib
import pickle
import yaml
import sys
import shutil

from framework.constants import ROOT_DIR
from framework.helpers.visualizer import plotMetricPerConfigPerLayer
from framework.helpers.design_metrics import calc_metric, SUPPORTED_METRICS
from framework.node.node_evaluator import LayerResult, DesignResult, NodeResult, NodeEvaluator

sys.path.append(str(pathlib.Path(ROOT_DIR, "tools", "zigzag")))
from tools.zigzag.zigzag.api import get_hardware_performance_zigzag
from tools.zigzag.zigzag.cost_model.cost_model import CostModelEvaluation

class Zigzag(NodeEvaluator):
    def __init__(self, in_config: dict) -> None:
        super().__init__()
        self.node_config = in_config
        self.config = in_config["evaluation"]
        self.fname_result = "zigzag_layers.csv"

        self.configs_dir = pathlib.Path(ROOT_DIR, "tools", "zigzag", "zigzag", "inputs")
        self.zigzag_hardware_configs_dir = pathlib.Path(self.configs_dir, "hardware")
        self.zigzag_mapping_configs_dir = pathlib.Path(self.configs_dir, "mapping")
        self.local_hardware_configs_dir = pathlib.Path(ROOT_DIR, "configs", "zigzag_configs", "hardware")
        self.local_mapping_configs_dir = pathlib.Path(ROOT_DIR, "configs", "zigzag_configs", "mapping")
        self.model_dir= pathlib.Path(ROOT_DIR, "onnx_models")

        self.accname = self.config["accelerator"] + ".yaml"
        self.mapname = self.config["mapping"] + ".yaml"
        self.optimization = self.config.get("optimization", "latency")
        assert self.optimization in ["energy", "latency", "EDP"]
        self.freq = self.config["frequency"]


    def set_workdir(self, work_dir: str, runname: str, id: int):
        self.runname = runname
        return super().set_workdir(work_dir, runname, id)

    def run(self, network: str, layers: list):
        design_runroot = pathlib.Path(self.runroot, "design"+str(1))
        zigzag_out_dir = pathlib.Path(design_runroot, "out")

        zigzag_pickle = pathlib.Path(zigzag_out_dir, "results.pkl")

        zigzag_config_dir_arch = pathlib.Path(design_runroot, "zigzag_config", "archs")
        zigzag_config_dir_arch.mkdir(parents=True, exist_ok=True)
        zigzag_config_dir_mapping = pathlib.Path(design_runroot, "zigzag_config", "mapping")
        zigzag_config_dir_mapping.mkdir(parents=True, exist_ok=True)

        # First, look in the ZigZag repository if the architecture file exists
        # Otherwise, check in the local folder
        accfile = pathlib.Path(self.zigzag_hardware_configs_dir, self.accname)
        mapfile = pathlib.Path(self.zigzag_mapping_configs_dir, self.mapname)
        if not accfile.exists():
            accfile = pathlib.Path(self.local_hardware_configs_dir, self.accname)
        if not mapfile.exists():
            mapfile = pathlib.Path(self.local_mapping_configs_dir, self.mapname)
        
        assert accfile.exists() and mapfile.exists(), f"Could not found the specified mapping or accelerator.\nAccfile: {str(accfile)}\nMapfile: {str(mapfile)}"

        modelfile = pathlib.Path(self.model_dir, self.runname, "new_model.onnx")

        shutil.copy(str(accfile), str(zigzag_config_dir_arch))
        shutil.copy(str(mapfile), str(zigzag_config_dir_mapping))

        #the model does not give area
        energy, latency, cmes = get_hardware_performance_zigzag(workload=str(modelfile), 
                                                                accelerator=str(accfile), 
                                                                mapping=str(mapfile), 
                                                                dump_folder=str(zigzag_out_dir), 
                                                                pickle_filename=str(zigzag_pickle),
                                                                opt=self.optimization)

        # Get layerwise results                                                
        node_result = NodeResult(self.node_config)
        design_result = DesignResult()
        layers = cmes[0][1]
        for layer in layers:
            cme = layer[0]
            l = LayerResult()
            l.name = cme.layer.name
            l.energy = cme.energy_total / 1e9 # pJ -> mJ
            l.latency = cme.latency_total2  / self.freq * 1e3 # cycles -> ms
            l.area = 0.0 # zigzag doesn't return the area

            design_result.add_layer(l)
            design_result.add_layer_to_network(l, network)

        node_result.add_design(design_result)
        
        # Convert to CNNParted stats
        self.stats = node_result.to_dict()
        return node_result


    def _run_layer(self):
        ...

    def _get_accelerator_config(self):
        ...

    def _get_workload(self):
        ...

