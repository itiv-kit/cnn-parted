from dataclasses import dataclass, asdict

@dataclass
class RobustnessConfig:
    bits: list[int]
    delta: float

    def __init__(self, cfg: dict):
        self.bits = cfg["bits"]
        self.delta = cfg["delta"]


@dataclass
class RetrainingConfig:
    epochs: int

    def __init__(self, cfg: dict):
        self.epochs = cfg.get("epochs", None)


@dataclass
class DatasetConfig:
    type: str
    kind: str
    path: str
    sample_limit: int | None
    batch_size: int
    randomize: bool
    file: str | None

    def __init__(self, cfg: dict):
        self.type = cfg["type"]
        self.kind = cfg["kind"]
        self.path = cfg["path"]
        self.sample_limit = cfg.get("sample_limit", None)
        self.batch_size = cfg["batch_size"]
        self.randomize = cfg.get("randomize", False)
        self.file = cfg.get("file", None)


@dataclass
class AccuracyConfig:
    robustness: RobustnessConfig
    retraining: RetrainingConfig
    calibrate: DatasetConfig
    validation: DatasetConfig
    train: DatasetConfig

    def __init__(self, cfg: dict):
        self.robustness = RobustnessConfig(cfg["robustness"])
        self.retraining = RetrainingConfig(cfg["retraining"])
        self.calibrate = DatasetConfig(cfg["datasets"]["calibrate"])
        self.validation = DatasetConfig(cfg["datasets"]["validation"])
        self.train = DatasetConfig(cfg["datasets"]["train"])
    
    def get_datasets_dict(self):
        return {"calibrate": asdict(self.calibrate),
                "validation": asdict(self.validation),
                "train": asdict(self.train)}
