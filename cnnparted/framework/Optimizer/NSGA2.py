from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# from pymoo.util.display.column import Column
# from pymoo.util.display.output import Output
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling #IntegerRandomSampling
from pymoo.factory import get_termination 
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from .Optimizer import Optimizer
import numpy as np

from .OptimizerHelper import OptimizerHelper 



class NSGA2_Optimizer(Optimizer):
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_gen = 20 * len(nodes) #10 time node size 
        self.pop_size = len(nodes) // 2 if len(nodes)>20 else len(nodes)
        self.opt_helper = OptimizerHelper()
        

    def optimize(self,optimization_objectives):
        problem= Problem(self.nodes)
        #bounds = [(1, len(self.nodes)-1)] 
        eta = 20
        mutation_prob = 0.1

        algorithm = NSGA2(
        pop_size=self.pop_size,
        n_offsprings=self.pop_size,
        sampling=FloatRandomSampling(),#In the beginning, initial points need to be sampled
        crossover=SBX(eta = eta,  prob=1),
        mutation=PM(eta = eta,prob= mutation_prob),
        eliminate_duplicates=True
        )

        res = minimize(problem,
               algorithm,
               termination=get_termination('n_gen',self.num_gen),
               seed=1,
               save_history=True,
               verbose=False
               )

        paretos = []
        X_rounded=np.floor(res.X)
        X_unique =np.unique(X_rounded, axis=0)
        for x in X_unique:
            paretos.append(int(x))
        
        opt_nodes=[]
        for node in paretos:
            opt_nodes.append(self.nodes[node])

        optimizer = self.opt_helper.find_best_node(opt_nodes,optimization_objectives)

        return optimizer,opt_nodes


class Problem(ElementwiseProblem):
    def __init__(self,data):

        sample_entry = data[next(iter(data))]

        super().__init__(n_var=1,
                         n_obj=len(sample_entry)-1, # -1:layer is not an objective, check  evaluator.get_all_layer_stats() 
                         n_constr=0,
                         xl=1, # not zero to ignore the first layer
                         xu=len(data)-1)# number of possible partioning points to be evaluated
        self.data = data
        
    def _evaluate(self, x, out, *args, **kwargs):
        idx = int(x.item())  # Convert numpy array to integer index

        objectives = [key for key in self.data[idx] if not key.endswith("_opt") and key not in ["layer"]]

        objectives_values = [self.data[idx][key] * self.data[idx].get(f"{key}_opt", 1) for key in objectives]

        out["F"] = np.array(objectives_values)
