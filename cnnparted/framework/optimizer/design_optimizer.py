import os
import pickle

import numpy as np
from numpy.random import default_rng

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from stable_baselines3 import PPO

from framework.optimizer.optimizer import Optimizer
from framework.optimizer.algorithms.rl_dse import DseEnv
from framework.node.node_evaluator import SystemResult, NodeResult, DesignResult, LayerResult

# This modules does the following:
#   - Instatiate GesignOptimizerGaProblem
#   - use NSGA2 to optimize
class DesignOptimizer(Optimizer):

    def __init__(self, node_components, problem, dse_config: dict, work_dir: str):
        self.num_gen = dse_config["optimizer"]["num_gen"]
        self.pop_size = dse_config["optimizer"]["pop_size"]

        self.node_components = node_components
        self.node_ids = [node["id"] for node in self.node_components]
        self.accelerator_names = [node["evaluation"]["accelerator"] for node in self.node_components]
        
        self.problem = problem
        self.algorithm = dse_config["optimizer"]["algorithm"]
        self.work_dir = work_dir

    def optimize(self, q_constr, conf):
        sorts = self._optimize_single()
        return sorts

    def _rl_wrapper(self, problem, algorithm, seed):
        ...

    def _gen_initial_x(self):
        samples = []
        rng = default_rng()
        dse_nodes = [node for node in self.node_components if "dse" in node]

        while(len(samples) < self.pop_size):
            sys_cfg = []
            for (node, constr) in zip(dse_nodes, self.problem.node_constraints):
                if "dse" in node:
                    xl, xu = constr[0], constr[1]
                    acc_cfg = rng.integers(low=xl, high=[x+1 for x in xu])
                    sys_cfg.append(acc_cfg)
            sys_cfg = np.array(sys_cfg).flatten().tolist()

            if sys_cfg not in samples:
                samples.append(sys_cfg)
        
        return np.array(samples)


    def _optimize_single(self):
        problem = self.problem

        initial_x = self._gen_initial_x()

        pymoo_algorithms = ["nsga2", "pso"]
        pydeap_algorithms = []
        rl_algorithms = ["rl_ppo"]

        if self.algorithm == "nsga2":
            algorithm = NSGA2(
                pop_size=self.pop_size,
                n_offsprings=self.pop_size,
                sampling=initial_x,
                crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
                mutation=PM(prob=0.9, eta=10, repair=RoundingRepair()),
                eliminate_duplicates=True
            )
        elif self.algorithm == "pso":
            algorithm = PSO(
                pop_size=self.pop_size,
                sampling=initial_x,
                repair=RoundingRepair()
            )
        elif self.algorithm == "rl_ppo":
            env = DseEnv(self.problem)
            algorithm = PPO('MlpPolicy', env, verbose=1, normalize_advantage=True, ent_coef=0.1, vf_coef= 0.5, n_steps=2046, batch_size=64) 
        else:
            raise RuntimeError("Invalid algorithm {self.algorithm}")
        
        # TODO This is a very hacky way to go about it, needs proper integration of
        #   - support only partial evaluation if only one node is present
        #   - handling of instances
        node_eval_stats = SystemResult()
        for node_id, accelerator_name in zip(self.node_ids, self.accelerator_names):
            file_str = str(node_id) + "_" + accelerator_name + "_tl_layers.csv"
            file_str = os.path.join(self.work_dir, file_str)
            file_str_alt = str(node_id) + "_tl_layers.csv"
            file_str_alt = os.path.join(self.work_dir, file_str_alt)
            if os.path.isfile(file_str):
                node_eval_stats.add_platform(NodeResult.from_csv(file_str))
            elif os.path.isfile(file_str_alt):
                node_eval_stats.add_platform(node_id, NodeResult.from_csv(file_str_alt))

        # For now, continue if everything is present
        if node_eval_stats.get_num_platforms() == len(self.node_components):
            return node_eval_stats.to_dict()

        # Select the corresponding wrapper to perform optimization
        if self.algorithm in pymoo_algorithms:
            res = minimize(problem,
                        algorithm,
                        termination=get_termination('n_gen', self.num_gen),
                        seed=1,
                        save_history=True,
                        verbose=False
                        )
            
            node_eval_stats = problem.system_results.to_dict()
            problem.system_results.to_csv(self.work_dir)
            self._plot_history(res, self.work_dir)

        elif self.algorithm in rl_algorithms:
            #rand_obs = algorithm.env.observation_space.sample()
            #action_before_training = algorithm.predict(rand_obs, deterministic=True)
            #algorithm.learn(total_timesteps=1000, progress_bar=False)
            breakpoint()
            env.step(env.action_space.sample())
            breakpoint()

            model_out_file = os.path.join(self.work_dir, "dse_results", "rl_model", "PPO_agent")               
            algorithm.save(model_out_file)

        # TODO: Return type currently does not consider res of minimize call
        return node_eval_stats
