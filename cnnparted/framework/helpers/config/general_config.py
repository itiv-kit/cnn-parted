from dataclasses import dataclass
from math import inf
from typing import Literal
from framework.model.tiling.iterators import LoopDim

@dataclass(init=False)
class GeneralConfig:
    fixed_sys: bool
    acc_once: bool
    max_num_platforms: int | None
    num_pp: int
    num_topos: int
    num_jobs: int
    optimization: Literal["all", "edp", "ppa"]
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
        self.work_dir = general_cfg.get("work_dir", None)
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

    def __init__(self, cfg: dict):
        self.name = cfg["name"]
        self.input_size = cfg["input-size"] 
        self.deadline = cfg.get("deadline", inf)
        self.frames = cfg.get("frames", 1)
        assert self.frames >= 1, "Frames must be at least 1"
        assert self.deadline >= 0, "Deadline must be non-negative"
        assert not (self.frames > 1 and self.deadline != inf), "Cannot set a deadline for a workload with multiple frames"
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
