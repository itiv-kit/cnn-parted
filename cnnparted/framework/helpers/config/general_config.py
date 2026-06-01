import os
from dataclasses import dataclass
from math import inf
from pathlib import Path

from framework.model.tiling.iterators import LoopDim

@dataclass(init=False)
class GeneralConfig:
    work_dir: str | None
    keep_dir: bool

    def __init__(self, general_cfg: dict):
        self.fixed_sys = general_cfg.get("fixed_sys", False)
        self.acc_once = general_cfg.get("acc_once", False)
        self.max_num_platforms = general_cfg.get("max_num_platforms", None)
        self.num_pp = general_cfg.get("num_pp", -1)
        self.num_topos = general_cfg.get("num_topos", 1)
        self.num_jobs = general_cfg.get("num_jobs", 1)
        self.optimization = general_cfg.get("optimization", "all")
        self.work_dir = general_cfg["work_dir"] #general_cfg.get("work_dir", None)
        self.plot_dir = os.path.join(self.work_dir, "figures")
        self.keep_dir = general_cfg.get("keep_dir", True)

@dataclass
class CnnpartedArguments:
    conf_file_path: str
    run_name: str
    show_progress: bool
    num_runs: int
    cuda: bool
    cuda_id: int
    clear_partitioning: bool
    clear_logs: bool
    top_k: int
    prune_strict: bool
    no_log: bool

    def __init__(self, args: dict):
        self.conf_file_path = args["conf_file_path"]
        self.run_name = args["run_name"]
        self.show_progress = args["p"]
        self.num_runs = args["num_runs"]
        self.cuda = args["cuda"]
        self.cuda_id = args["cid"]
        self.clear_partitioning = args["clear_partitioning"]
        self.clear_logs = args["clear_logs"]
        self.top_k = args["top_k"]
        self.no_log = args["no_log"]
        self.prune_strict = args["prune_strict"]


### Everything related to the workload configuration

class TilingConfig:
    SUPPORTED_OP_TYPES = ["Conv", "Gemm", "MatMul"]
    SUPPORTED_TILING_DIMS = [LoopDim.N, LoopDim.M, LoopDim.P, LoopDim.Q]

    def __init__(self, cfg: dict):
        self.tiling: dict[LoopDim, int] = {
            LoopDim.N: cfg.get("N"),
            LoopDim.M: cfg.get("M"),
            LoopDim.P: cfg.get("P"),
            LoopDim.Q: cfg.get("Q"),
        }

    @property
    def num_tiles(self):
        return [num for num in self.tiling.values()]
    
    def __iter__(self):
        return iter(self.tiling.items())

@dataclass
class WorkloadSpec:
    name: str
    input_size: list[int]
    tiling: dict[str, TilingConfig] #op_type -> config
    deadline: int
    frames: int
    deadline_mul: float
    deadline_div: float

    def __init__(self, cfg: dict):
        self.input_size = cfg["input-size"] 

        # Parse the name of the file
        name_path = Path(cfg["name"])
        # Check if the name points to an ONNX file or is a PyTorch model
        if name_path.is_file() and name_path.suffix == ".onnx":
            self.src_model_fname = str(name_path)
            self.dest_model_fname = name_path.stem + "_model.onnx"
            self.name = name_path.stem
        else:
            self.src_model_fname = str(name_path) #Model is generated from onnx, no src directory
            self.dest_model_fname = str(name_path) + "_model.onnx"
            self.name = name_path.stem

        # Deadline related configuration
        self.deadline = cfg.get("deadline", inf)
        self.frames = cfg.get("frames", None)
        self.deadline_mul = cfg.get("deadline_mul", None)
        self.deadline_div = cfg.get("deadline_div", None)
        assert self.deadline >= 0, "Deadline must be non-negative"
        assert not (self.frames is not None and self.deadline != inf), "Cannot set a deadline for a workload with multiple frames"
        use_auto_deadline = self.deadline != inf and (self.deadline_div is not None and self.deadline_mul is not None)
        assert not (use_auto_deadline and (self.deadline_div is None or self.deadline_mul is None)), "Auto-deadline requires specifying dealine_mul AND deadline_div"
        assert not (use_auto_deadline and (self.deadline_div > 0.0 and self.deadline_mul > 0.0)), "Deadline scaling cannot be negative"

        tiling_cfg = cfg.get("tiling", {})
        self.tiling = {}
        for op_type in TilingConfig.SUPPORTED_OP_TYPES:
            self.tiling[op_type] = TilingConfig(tiling_cfg.get(op_type, {}))


@dataclass(init=False)
class WorkloadConfig:
    workload_spec: list[WorkloadSpec]

    def __init__(self, workload_cfg: list[dict]):
        self.workload_spec = []
        for cfg in workload_cfg:
            self.workload_spec.append(
                    WorkloadSpec(cfg)
                )

@dataclass
class PipelineConfig:
    stages: list[str]

    def __init__(self, cfg: dict):
        self.stages = cfg["stages"]
