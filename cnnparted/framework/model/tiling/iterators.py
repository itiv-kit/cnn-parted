from enum import Enum, auto


class IteratorKind(Enum):
    PARALLEL = auto()
    REDUCTION = auto()


class LoopIterator:
    """An iterator represents a loop dimension"""
    def __init__(self, name: str, kind: IteratorKind, aliases: list[str]):
        self.name = name
        self.kind = kind
        self.aliases = aliases
    
    def __eq__(self, value):
        if not isinstance(value, LoopIterator):
            return False
        return self.kind == value.kind and \
            (self.name == value.name or 
             self.name in value.aliases or
             (value.name in self.aliases))

class LoopDim(Enum):
    N = LoopIterator("N", IteratorKind.PARALLEL, aliases=[])  #batch size conv
    B = LoopIterator("B", IteratorKind.PARALLEL, aliases=[])  #batch size matmul
    P = LoopIterator("P", IteratorKind.PARALLEL, aliases=["Ow"]) #output width 
    Q = LoopIterator("Q", IteratorKind.PARALLEL, aliases=["Oh"]) #output height
    R = LoopIterator("R", IteratorKind.REDUCTION,aliases=["Kw"]) #kernel width
    S = LoopIterator("S", IteratorKind.REDUCTION,aliases=["Kh"]) #kernel height
    C = LoopIterator("C", IteratorKind.REDUCTION,aliases=["Ic"]) #input channel
    M = LoopIterator("M", IteratorKind.PARALLEL, aliases=["Oc", "K"]) #output channel


class ConcreteLoopIterator():
    """Iterator with defined loop boundaries with iterator in [lower_bound, upper_bound)"""
    def __init__(self, iterator: LoopIterator, extent: int, 
                 lower_bound: int = None, upper_bound: int = None, 
                 postfix: str = ""):
        self.iterator = iterator
        self.extent = extent
        self.lower_bound = lower_bound if lower_bound is not None else 0        #lower bound is inclusive
        self.upper_bound = upper_bound if upper_bound is not None else extent-1 #upper bound is exclusive
        self.postfix = postfix
    
    def tile(self, factors: list[int]):
        """Create two new iterators by applying tiling"""
        assert len(factors) == 2, "Tiling expects factors in the form of (outer, inner)"

        outer, inner = factors
        if outer is not None and inner is not None:
            assert outer*inner == self.extent, "outer*inner must be equal to old loop extent"
        
        # Allow one of the values to be None, in this case automatically deduce the factor
        if outer is None:
            outer = self.extent // inner
        if inner is None:
            inner = self.extent // outer

        new_outer_it = ConcreteLoopIterator(self.iterator, outer, postfix="_outer")
        new_inner_it = ConcreteLoopIterator(self.iterator, inner, postfix="_inner")
        return new_outer_it, new_inner_it



    