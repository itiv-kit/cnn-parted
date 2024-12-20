
class OptimizerConfig:
    def __init__(self):
        # Number of variables, objectives and constraints
        self.n_var = None
        self.n_obj = None
        self.n_constr = None

        # Boundaries on values
        self.xl = None
        self.xu = None