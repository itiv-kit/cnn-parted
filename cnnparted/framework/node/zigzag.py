import pathlib
import pickle
from ruamel.yaml import YAML
import importlib
import sys
import shutil
import os
from loguru import logger

from framework.constants import ROOT_DIR
from framework.dse.interfaces.zigzag_adaptor import ZigzagAdaptor
from framework.model.onnx_model import OnnxModel
from framework.dse.interfaces.architecture_config import ArchitectureConfig
from framework.node.node_evaluator import LayerResult, DesignResult, NodeResult, NodeEvaluator
from framework.helpers.config import NodeConfig, EvaluationConfig

sys.path.append(str(pathlib.Path(ROOT_DIR, "cnnparted", "tools", "zigzag")))
from tools.zigzag.zigzag.api import get_hardware_performance_zigzag

yaml = YAML(typ="rt")

class Zigzag(NodeEvaluator):
    def __init__(self, in_config: NodeConfig, run_name: str) -> None:
        super().__init__()
        self.node_config = in_config
        self.dse_node_config = in_config.dse
        self.config: EvaluationConfig = in_config.evaluation
        self.run_name = run_name
        self.fname_result = "zigzag_layers"

        self.zigzag_in_configs_dir_local = os.path.join(ROOT_DIR, "configs", "zigzag_configs")
        self.zigzag_in_configs_dir_repo = os.path.join(ROOT_DIR, "cnnparted", "tools", "zigzag", "zigzag", "inputs")

        self.configs_dir = pathlib.Path(ROOT_DIR, "tools", "zigzag", "zigzag", "inputs")
        self.zigzag_hardware_configs_dir = pathlib.Path(self.configs_dir, "hardware")
        self.zigzag_mapping_configs_dir = pathlib.Path(self.configs_dir, "mapping")
        self.local_hardware_configs_dir = pathlib.Path(ROOT_DIR, "configs", "zigzag_configs", "hardware")
        self.local_mapping_configs_dir = pathlib.Path(ROOT_DIR, "configs", "zigzag_configs", "mapping")
        self.model_dir= pathlib.Path(ROOT_DIR, "onnx_models")

        self.accname = self.config.accelerator + ".yaml"
        self.mapname = self.config.simulator_cfg.mapping + ".yaml"
        self.optimization = self.config.simulator_cfg.optimization
        assert self.optimization in ["energy", "latency", "EDP"]
        self.freq = self.config.frequency

        self.adaptor: ZigzagAdaptor = None
        self.cur_design_id = 0
        if self.dse_node_config is not None and self.dse_node_config.mutator is not None:

            dse_package = importlib.import_module("framework.dse")
            mutator_cls = getattr(dse_package, self.dse_node_config.mutator)
            self.adaptor = mutator_cls()

            if self.dse_system_config.optimizer.algorithm == "exhaustive": #== "exhaustive":
                self.adaptor.read_space_cfg(self.dse_node_config)
            elif self.dse_system_config.optimizer.algorithm == "nsga2":
                pass

    def _find_hardware_config(self, config_file: str | pathlib.Path):
        if (local_path := pathlib.Path(self.zigzag_in_configs_dir_local, "hardware", config_file)).exists():
            return local_path
        elif(repo_path := pathlib.Path(self.zigzag_in_configs_dir_repo, "hardware", config_file)).exists(): 
            return repo_path

    def _find_mapping_config(self, config_file: str | pathlib.Path):
        if (local_path := pathlib.Path(self.zigzag_in_configs_dir_local, "mapping", config_file)).exists():
            return local_path
        elif(repo_path := pathlib.Path(self.zigzag_in_configs_dir_repo, "mapping", config_file)).exists(): 
            return repo_path

    def set_workdir(self, work_dir: str, runname: str, id: int):
        return super().set_workdir(work_dir, runname, id)

    def run(self, network: str, model: OnnxModel) -> NodeResult:
        design_runroot = pathlib.Path(self.runroot)
        node_result = NodeResult(self.node_config)
        self._run_network(network, model, design_runroot, node_result, design=None)
        return node_result

    def run_from_adaptor(self, network: str, model : OnnxModel, adaptor: ZigzagAdaptor, progress : bool = False) -> NodeResult:
        self.adaptor = adaptor
        node_result = NodeResult(self.node_config)

        if not self.adaptor.design_space:
            self._run_design(network, model, progress, node_result, adaptor.config, self.cur_design_id)
            self.cur_design_id += 1

            self.stats = node_result.to_dict()
            self.adaptor = None
            return node_result
        
        else:
            print(f"There are a total of {len(self.adaptor.design_space)} designs to be evaluated!")
            
            for i, design in enumerate(self.adaptor.design_space):
                if os.path.exists(os.path.join(self.runroot, "design"+str(i))):
                    shutil.rmtree(os.path.join(self.runroot, "design"+str(i)))
                self._run_design(network, model, progress, node_result, design, i)
            
        self.stats = node_result.to_dict()
        with open(os.path.join(self.runroot, f"results_{self.accname}.pkl"), "wb") as f:
            pickle.dump(self.stats, f)

        return node_result

    def _run_design(self, network: str, model: OnnxModel, progress: bool, node_result: NodeResult,  design: ArchitectureConfig, design_id: int=None):
        if self.adaptor and self.adaptor.zigzag_out_configs_dir:
            design_runroot = os.path.join(self.runroot, self.adaptor.zigzag_out_configs_dir)
        else:
            if design_id is None:
                design_id = 0
            design_runroot = os.path.join(self.runroot, "design"+str(design_id))
        zigzag_design_dir = os.path.join(design_runroot, "zigzag_config")
        zigzag_design_dir_hw = os.path.join(design_runroot, "zigzag_config", "hardware")
        zigzag_design_dir_mapping = os.path.join(design_runroot, "zigzag_config", "mapping")
        os.makedirs(zigzag_design_dir_hw, exist_ok=True)
        os.makedirs(zigzag_design_dir_mapping, exist_ok=True)

        self.adaptor.run_zigzag_from_config(design, outdir=zigzag_design_dir)
        with open(os.path.join(design_runroot, "arch_config.yaml"), "w") as f:
            _ = yaml.dump(design.get_config(), f)

        self._run_network(network, model, design_runroot, node_result, design)

    def _run_network(self, network: str, model: OnnxModel, zigzag_design_workdir: str, node_result: NodeResult, design: ArchitectureConfig = None):
        accfile = self._find_hardware_config(self.accname)
        mapfile = self._find_mapping_config(self.mapname)
        assert accfile.exists() and mapfile.exists(), f"Could not found the specified mapping or accelerator.\nAccfile: {str(accfile)}\nMapfile: {str(mapfile)}"
        
        zigzag_design_workdir_hardware = pathlib.Path(zigzag_design_workdir, "hardware")
        zigzag_design_workdir_mapping = pathlib.Path(zigzag_design_workdir, "mapping")
        os.makedirs(zigzag_design_workdir_hardware, exist_ok=True)
        os.makedirs(zigzag_design_workdir_mapping, exist_ok=True)
        shutil.copy(str(accfile), zigzag_design_workdir_hardware)
        shutil.copy(str(mapfile), zigzag_design_workdir_mapping)

        zigzag_out_dir = pathlib.Path(zigzag_design_workdir, "out")
        zigzag_pickle = pathlib.Path(zigzag_design_workdir, "out", "results.pkl")

        #modelfile = pathlib.Path(self.model_dir, self.run_name, network + "_new_model.onnx")
        logger.debug(f"Zigzag: Running model {network}...")
        logger.debug(f"Zigzag: Accelerator from {accfile}")
        logger.debug(f"Zigzag: Mapping from {accfile}")

        #the model does not give area
        energy, latency, cmes = get_hardware_performance_zigzag(workload=model.onnx_in_model, 
                                                                accelerator=str(accfile), 
                                                                mapping=str(mapfile), 
                                                                dump_folder=str(zigzag_out_dir), 
                                                                pickle_filename=str(zigzag_pickle),
                                                                opt=self.optimization)

        if design is not None:                                         
            design_result = DesignResult(design.get_config())
        else:
            design_result = DesignResult()

        # Get layerwise results
        layers = cmes[0][1]
        for layer in layers:
            cme = layer[0]
            #TODO Is this check necessary?
            if cme.layer.type in ["Conv", "Gemm", "MatMul"]:
                lyr = LayerResult()
                lyr.name = cme.layer.name
                lyr.energy = cme.energy_total / 1e9 # pJ -> mJ
                lyr.latency = cme.latency_total2  / self.freq * 1e3 # cycles -> ms
                lyr.area = 0.0 # zigzag doesn't return the area

                design_result.add_layer_to_network(lyr, network)

        node_result.add_design(design_result)
        
