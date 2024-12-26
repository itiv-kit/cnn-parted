import numpy as np
from pymoo.core.mutation import Mutation

class PartitioningMutation(Mutation):
    def __init__(self, num_layers, prob=0.9, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.num_layers = num_layers

    def _do(self, problem, X, **kwargs):
        assert problem.node_stats is not None
        num_platforms = len(problem.node_stats)

        X = X.astype(object)

        for i in range(len(X)):
            mut = np.where(np.random.random(len(X[i])) < self.prob.value)[0]

            for j in mut:
                if j < (X.shape[1] / 2 - 1):
                    X[i, j] = np.random.randint(1, self.num_layers + 1)
                else:
                    X[i, j] = np.random.randint(1, num_platforms + 1)

        X = X.astype(int)

        return X