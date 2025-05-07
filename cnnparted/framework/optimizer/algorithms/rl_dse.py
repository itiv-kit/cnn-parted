import numpy as np
import gymnasium as gym
from gymnasium import spaces

from framework.optimizer.design_problem import DesignProblem

class DseEnv(gym.Env):
    def __init__(self, design_problem: DesignProblem):
        super().__init__()
        self.design_problem = design_problem

        xl = self.design_problem.xl
        xu = self.design_problem.xu
        action_space_vec = xu - xl + 1 #TODO: Double Check

        self.max_area = None
        self.max_latency = None
        self.max_energy = None

        self.action_space = spaces.MultiDiscrete(action_space_vec, start=xl)

        #[latency, energy, throughput, area, link_latency, link_energy]
        self.obs_space_low = np.array([0, 0, 0, 0, 0, 0])
        self.obs_space_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=self.obs_space_low, high=self.obs_space_high)
        self.current_obs = None

    def step(self, action):
        terminated = True
        truncated = False
        
        # Evaluate the sampled design and perform partitioning
        out = {}
        self.design_problem._evaluate(action.tolist(), out)

        observation = out["G"][-6:]

        reward = -out["F"] #cost function

        return observation, reward, terminated, truncated, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        initial_observation = np.array([1, 2, 3, 4], dtype=np.float32)
        return initial_observation, {}
    
    def render(self):
        ...

    def close(self):
        ...
