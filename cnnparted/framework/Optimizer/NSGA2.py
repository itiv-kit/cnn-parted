from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# from pymoo.util.display.column import Column
# from pymoo.util.display.output import Output
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling #IntegerRandomSampling
#from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from .Optimizer import Optimizer
import numpy as np
from .OptimizerHelper import OptimizerHelper 

class NSGA2_Optimizer(Optimizer):
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_gen = 500 # needs to be problem specific
        self.pop_size = 30 # same
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
        mutation=PM(prob= mutation_prob),
        eliminate_duplicates=True
        )

        #termination = get_termination("n_gen", ..)

        res = minimize(problem,
               algorithm,
               termination=('n_gen',self.num_gen),
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

        return optimizer


class Problem(ElementwiseProblem):
    def __init__(self,data):
        super().__init__(n_var=1,
                         n_obj=9,
                         n_constr=0,
                         xl=1, # not zero to ignore the first layer
                         xu=len(data)-1)# number of partioning points to be evaluated
        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        idx = int(x.item())  # Convert numpy array to integer index
        latency_objective = self.data[idx]['latency']
        energy_objective = self.data[idx]['energy']
        sensor_latency_objective = self.data[idx]['sensor_latency']
        sensor_energy_objective = self.data[idx]['sensor_energy']
        link_latency_objective = self.data[idx]['link_latency']
        link_energy_objective = self.data[idx]['link_energy']
        edge_latency_objective = self.data[idx]['edge_latency']
        edge_energy_objective = self.data[idx]['edge_energy']
        goodput_objective = -1 * self.data[idx]['throughput']  # needs to be maximized (* -1)

        out["F"] = np.array([   latency_objective,
                                energy_objective,
                                sensor_latency_objective,
                                sensor_energy_objective,
                                link_latency_objective,
                                link_energy_objective,
                                edge_latency_objective,
                                edge_energy_objective,
                                goodput_objective])