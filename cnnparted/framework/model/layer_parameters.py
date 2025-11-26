from dataclasses import dataclass, field
from enum import IntEnum, auto
from abc import ABC, abstractmethod
import numpy as np

class ParametersBase(ABC):
    @abstractmethod
    def get_input_tensor_shapes(self):
        ...

    @abstractmethod
    def get_input_tensor_dimensions(self):
        ...

    @abstractmethod
    def get_output_tensor_shape(self):
        ...

    @abstractmethod
    def get_output_tensor_dimensions(self):
        ...

    @property
    @abstractmethod
    def dimensions(self):
        ...

@dataclass
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
        self.depthwise = False #TODO Figure this out

    @property
    def dimensions(self):
        return ["N", "M", "P", "Q", "C", "R", "S"]
    
    def get_input_tensor_shapes(self):
        input_shape = [self.n, self.c, self.h, self.w]
        kernel_shape = [self.m, self.c, self.r, self.s]
        bias_shape = [self.m]
        return [input_shape, kernel_shape, bias_shape]

    def get_input_tensor_dimensions(self):
        #(N,Cin​,H,W) and output (N,Cout,Hout,Wout)
        # Kernel: (M, C/groups, R, S)
        input_dimensions = ["N", "C", "H", "W"]
        kernel_dimensions = ["M", "C", "R", "S"]
        bias_dimensions = ["M"]
        return [input_dimensions, kernel_dimensions, bias_dimensions]
    
    def get_output_tensor_dimensions(self):
        return [["N", "M", "P", "Q"]]
    
    def get_output_tensor_shape(self):
        oshape = [self.n, self.c, self.p, self.q]
        return [oshape]
    

@dataclass
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
    
    @property
    def dimensions(self):
        dims = ["N", "M", "C"]
        if self.b is not None:
            dims = ["B"] + dims
        return dims
    
    def get_input_tensor_shapes(self):
        input_shape = [self.n, self.c]
        weight_shape = [self.c, self.m]
        bias_shape = [self.n, self.m]
        out = [input_shape, weight_shape, ] #bias_shape
        if self.has_bias:
            out.append(bias_shape)
        return out
    
    def get_input_tensor_dimensions(self):
        out = [["N", "C"], ["C", "M"], ]
        if self.has_bias:
            out.append(["N", "M"])
        return out

    def get_output_tensor_dimensions(self):
        return [["N", "M"]]
    
    def get_output_tensor_shape(self):
        oshape = [self.n, self.m]
        return [oshape]
    
    
@dataclass
class PoolingParameters(ParametersBase):
    kernel_shape: np.ndarray
    padding: np.ndarray | None
    stride: np.ndarray

    def __init__(self, cfg: dict):
        self.kernel_shape = cfg["kernel"]
        self.padding = cfg["padding"]
        self.stride = cfg["stride"]
    
    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [[]]

    def get_input_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_dimensions(self):
        return [[]]
    
    def get_output_tensor_shape(self):
        return [[]]

class ActivationKind(IntEnum):
    Relu = auto()
    Sigmoid = auto()
    Erf = auto()
    unknown = auto()

@dataclass
class ActivationParameters(ParametersBase):
    shape: list[int]
    kind: ActivationKind

    def __init__(self, cfg: dict):
        self.shape = cfg["shape"]
        self.kind = ActivationKind[cfg["kind"]]
    
    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [self.shape]
    
    def get_input_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_dimensions(self):
        return [[]]
    
    def get_output_tensor_shape(self):
        return [self.shape]

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
    
    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [self.shape,  self.shape]
    
    def get_input_tensor_dimensions(self):
        return [[], []]

    def get_output_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_shape(self):
        return [self.shape]

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

    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [self.in_shape]
    
    def get_input_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_shape(self):
        return [self.out_shape]

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

    @property
    def dimensions(self):
        return []

    def get_input_tensor_shapes(self):
        return [self.a_shape, self.b_shape]
    
    def get_input_tensor_dimensions(self):
        return [[], []]

    def get_output_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_shape(self):
        return [self.out_shape]

@dataclass
class ConstantOperationParameters:
    kind: MiscOperationsKind

    def __init__(self):
        self.kind = MiscOperationsKind.Constant

    @property
    def dimensions(self):
        return []

    def get_input_tensor_shapes(self):
        return []
    
    def get_input_tensor_dimensions(self):
        return []

    def get_output_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_shape(self):
        return [[]]


@dataclass
class TensorParameters:
    name: str
    shape: list[int]
    #elem_type: np.integer | np.floating
    definition: str = None
    used_by: list[str] = field(default_factory=list)
    dimensions: list[str] = None

    @property
    def num_elems(self):
        return np.prod(self.shape)

type ComputeParameters = ConvParameters | GemmParameters
type AnyLayerParameters = ComputeParameters | PoolingParameters | ActivationParameters | ElementwiseParameters | BinaryMiscOperationParameters | UnaryMiscOperationParameters

@dataclass
class LayerParameters:
    # A layer consists of the computational parameters and the input and output tensors
    name: str
    inputs: list[str]
    op_type: str
    outputs: list[str]
    output_size: np.ndarray
    params: AnyLayerParameters | None
    dimensions: list[str]
    children: list[str]

    def __init__(self, cfg: dict, params: ComputeParameters | None):
        self.name = cfg["name"]
        self.op_type = cfg["op_type"]

        if isinstance(cfg["input"], list):
            self.inputs = cfg["input"]
        else:
            self.inputs = [cfg["input"]]

        if isinstance(cfg["output"], list):
            self.outputs = cfg["output"]
        else:
            self.outputs = [cfg["output"]]

        if len(cfg["output_size"]) == 0 and cfg["op_type"] == "Constant":
            self.output_size = [[]]
        elif isinstance(cfg["output_size"][0], list):
            self.output_size = cfg["output_size"]
        else:
            self.output_size = [cfg["output_size"]]

        self.params = params
        if params is not None:
            self.dimensions = params.dimensions
        else:
            self.dimensions = None
        
        self.children = None
