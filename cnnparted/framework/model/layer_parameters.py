from dataclasses import dataclass, field
from enum import IntEnum, auto
from abc import ABC
import numpy as np

class ParametersBase(ABC):
    ...

@dataclass(unsafe_hash=True)
class ConvParameters(ParametersBase):
    h: int
    w: int
    n: int
    m: int
    q: int
    p: int
    c: int
    s: int
    r: int
    wpad: int
    hpad: int
    wstride: int
    hstride: int
    ifms: int
    ofms: int
    weights: int
    groups: int
    wdilation: int
    hdilation: int
    depthwise: bool

    def __init__(self, cfg: dict):
        self.h = cfg["h"]
        self.w = cfg["w"]
        self.n = cfg["n"]
        self.m = cfg["m"]
        self.q = cfg["q"]
        self.p = cfg["p"]
        self.c = cfg["c"]
        self.s = cfg["s"]
        self.r = cfg["r"]
        self.wpad = cfg["wpad"]
        self.hpad = cfg["hpad"]
        self.wstride = cfg["wstride"]
        self.hstride = cfg["hstride"]
        self.ifms = cfg["ifms"]
        self.ofms = cfg["ofms"]
        self.weights = cfg["weights"]
        self.groups = cfg["groups"]
        self.wdilation = cfg["wdilation"]
        self.hdilation = cfg["hdilation"]
        self.depthwise = (self.groups == self.c) and (self.m % self.c == 0)


@dataclass(unsafe_hash=True)
class GemmParameters(ParametersBase):
    b: int | None
    n: int
    m: int
    c: int
    has_bias: bool
    weights: int

    def __init__(self, cfg: dict):
        self.b = cfg.get("b")
        self.n = cfg["n"]
        self.m = cfg["m"]
        self.c = cfg["c"]
        self.weights = cfg["weights"]
        self.has_bias = cfg["has_bias"]
    
    
@dataclass
class PoolingParameters(ParametersBase):
    kernel_shape: np.ndarray
    padding: np.ndarray | None
    stride: np.ndarray

    def __init__(self, cfg: dict):
        self.kernel_shape = cfg["kernel"]
        self.padding = cfg["padding"]
        self.stride = cfg["stride"]
    

class ActivationKind(IntEnum):
    Relu = auto()
    Sigmoid = auto()
    Erf = auto()
    HardSwish = auto()
    HardSigmoid = auto()
    unknown = auto()

@dataclass
class ActivationParameters(ParametersBase):
    shape: list[int]
    kind: ActivationKind

    def __init__(self, cfg: dict):
        self.shape = cfg["shape"]
        self.kind = ActivationKind[cfg["kind"]]
    

class ElementwiseKind(IntEnum):
    Add = auto()
    Mul = auto()
    Div = auto()
    unknown = auto()

@dataclass
class ElementwiseParameters(ParametersBase):
    shape: list[int]
    kind: ElementwiseKind

    def __init__(self, cfg: dict):
        self.shape = cfg["shape"]
        self.kind = ElementwiseKind[cfg["kind"]]
    

class MiscOperationsKind(IntEnum):
    Flatten = auto()
    GlobalAveragePool = auto()
    Concat = auto()
    Transpose = auto()
    LayerNormalization = auto()
    Constant = auto()
    Clip = auto()
    Shape = auto()
    Slice = auto()
    Resize = auto()
    unknown = auto()

@dataclass
class UnaryMiscOperationParameters:
    in_shape: list[int]
    out_shape : list[int]
    kind: MiscOperationsKind

    def __init__(self, cfg: dict):
        self.in_shape = cfg["in_shape"]
        self.out_shape = cfg["out_shape"]
        self.kind = MiscOperationsKind[cfg["kind"]]

@dataclass
class BinaryMiscOperationParameters:
    a_shape: list[int]
    b_shape: list[int]
    out_shape = list[int]
    kind: MiscOperationsKind

    def __init__(self, cfg: dict):
        self.a_shape = cfg["a_shape"]
        self.b_shape = cfg["b_shape"]
        self.out_shape = cfg["out_shape"]
        self.kind = MiscOperationsKind[cfg["kind"]]


@dataclass
class ConstantOperationParameters:
    kind: MiscOperationsKind

    def __init__(self):
        self.kind = MiscOperationsKind.Constant


#@dataclass
#class TensorParameters:
#    name: str
#    shape: list[int]
#    #elem_type: np.integer | np.floating
#    definition: str = None
#    used_by: list[str] = field(default_factory=list)
#    dimensions: list[str] = None
#
#    @property
#    def num_elems(self):
#        return np.prod(self.shape)

type ComputeParameters = ConvParameters | GemmParameters
type AnyLayerParameters = ComputeParameters | PoolingParameters | ActivationParameters | ElementwiseParameters | BinaryMiscOperationParameters | UnaryMiscOperationParameters

#@dataclass
#class LayerParameters:
#    # A layer consists of the computational parameters and the input and output tensors
#    name: str
#    inputs: list[str]
#    op_type: str
#    outputs: list[str]
#    output_size: np.ndarray
#    params: AnyLayerParameters | None
#    dimensions: list[str]
#    children: list[str]
#
#    def __init__(self, cfg: dict, params: ComputeParameters | None):
#        self.name = cfg["name"]
#        self.op_type = cfg["op_type"]
#
#        if isinstance(cfg["input"], list):
#            self.inputs = cfg["input"]
#        else:
#            self.inputs = [cfg["input"]]
#
#        if isinstance(cfg["output"], list):
#            self.outputs = cfg["output"]
#        else:
#            self.outputs = [cfg["output"]]
#
#        if len(cfg["output_size"]) == 0 and cfg["op_type"] == "Constant":
#            self.output_size = [[]]
#        elif isinstance(cfg["output_size"][0], list):
#            self.output_size = cfg["output_size"]
#        else:
#            self.output_size = [cfg["output_size"]]
#
#        self.params = params
#        if params is not None:
#            self.dimensions = params.dimensions
#        else:
#            self.dimensions = None
#        
#        self.children = None
