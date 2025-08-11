import yaml
from dataclasses import dataclass
from typing import Literal, TypedDict

@dataclass(init=False)
class GeneralConfig:
    fixed_sys: bool
    acc_once: bool
    num_pp: int
    num_topos: int
    num_jobs: int
    optimization: Literal["all", "edp", "ppa"]
    work_dir: str
    keep_dir: bool

    def __init__(self, general_cfg: dict):
        self.fixed_sys = general_cfg.get("fixed_sys", False)
        self.acc_once = general_cfg.get("acc_once", False)
        self.num_pp = general_cfg.get("num_pp", -1)
        self.num_topos = general_cfg.get("num_topos", 1)
        self.num_jobs = general_cfg.get("num_jobs", 1)
        self.optimization = general_cfg.get("optimization", "all")
        self.work_dir = general_cfg.get("work_dir", None)
        self.keep_dir = general_cfg.get("keep_dir", True)

### Everything related to the workload configuration
@dataclass
class WorkloadSpec:
    name: str
    input_size: list[int]


@dataclass(init=False)
class WorkloadConfig:
    workload_spec: list[WorkloadSpec]

    def __init__(self, workload_cfg: list[dict]):
        self.workload_spec = []
        for cfg in workload_cfg:
            self.workload_spec.append(
                    WorkloadSpec(cfg["name"], cfg["input-size"])
                )

@dataclass
class PipelineConfig:
    stages: list[str]

    def __init__(self, cfg: dict):
        self.stages = cfg["stages"]
