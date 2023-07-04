from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from .Optimizer import Optimizer
import numpy as np

class NSGA2_Optimizer(Optimizer):
    def __init__(self, nodes, num_generations, population_size):
        self.nodes = nodes
        self.num_gen = num_generations
        self.pop_size = population_size
        

    def optimize(self):
        problem= Problem(self.nodes)
        bounds = [(1, len(self.nodes)-1)] 
        eta = 20
        mutation_prob = 0.1

        algorithm = NSGA2(
        pop_size=self.pop_size,
        n_offsprings=self.pop_size,
        sampling=IntegerRandomSampling(),#In the beginning, initial points need to be sampled
        crossover=SBX(eta = eta,  prob=1),
        mutation=PM(prob= mutation_prob),
        #output = MyOutput(),
        eliminate_duplicates=True
        )

        termination = get_termination("n_gen", self.num_gen)

        #t0 = time.time()
        res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
    
        #t1 = time.time()
        #t = t1 - t0
        X = res.X
        F = res.F

    #print(F)
    #print(X)
        output = []
        X_rounded=np.floor(X)
        X_unique =np.unique(X_rounded, axis=0)
        for x in X_unique:
            print(int(x))
            output.append(int(x))
        #for f in F:
         #   print(f)
        return output


class Problem(ElementwiseProblem):
    def __init__(self,data):
        super().__init__(n_var=1,
                         n_obj=3,
                         n_constr=0,
                         xl=1, # not zero to ignore the first layer
                         xu=len(data)-1)# number of partioning points to be evaluated
        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        idx = int(x.item())  # Convert numpy array to integer index
        latency_objective = self.data[idx]['latency']
        energy_objective =  self.data[idx]['energy']
        goodput_objective = -1* self.data[idx]['goodput']# needs to be maximized(*-1)
        # TODO : add other objectives


        out["F"] = np.array([ latency_objective, energy_objective,goodput_objective])


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.points = Column("Best points")
        self.columns += [self.points]


    def update(self, algorithm):
        super().update(algorithm)
        self.points = algorithm.pop.get("X")

