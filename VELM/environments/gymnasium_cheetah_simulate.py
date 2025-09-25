import copy
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# This utility would contain the function to run inference with a learned dynamics model.
# Since we don't have the actual file, we'll assume it has a function:
# eval_model(model, state, action, stds) -> next_state
from . import simulated_env_util


class Gymnasium_CheetahSimulateEnv(gym.Env):
    """
    This class provides a simulated version of the Cheetah-v4 environment.
    It uses a learned dynamics model to predict the next state instead of
    relying on the underlying physics engine.
    """
    def __init__(self, learned_model: List[Any] = [], stds: List[float] = []):
        super(Gymnasium_CheetahSimulateEnv, self).__init__()

        # --- Define spaces consistent with the real Cheetah-v4 environment ---
        # Action space: 6-dimensional vector for joint torques
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # Observation space: 17-dimensional state vector for Cheetah-v4
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)

        # Initial state space (a small box around a neutral pose)
        low_init = -0.1 * np.ones(17)
        high_init = 0.1 * np.ones(17)
        self.init_space = gym.spaces.Box(low=low_init, high=high_init, dtype=np.float64)

        self.rng = np.random.default_rng()

        # Store the learned model and standard deviations for normalization
        self.model = copy.deepcopy(learned_model)
        self.stds = copy.deepcopy(stds)

        self._max_episode_steps = 1000
        self.state = None
        self.steps = 0
        
        # Timestep (dt) from the original Cheetah-v4 environment (5 frames * 0.01s)
        self.dt = 0.05
        
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the simulated environment to a random initial state.
        """
        if seed is not None:
            self.seed(seed)
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        """
        Advances the simulation using the learned model.
        """
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")

        # Predict the next state using the learned dynamics model
        self.state = simulated_env_util.eval_model(self.model, self.state, action, stds=self.stds)

        # --- Replicate Cheetah-v4's reward function ---
        # x_velocity is the 8th element of the observation space
        x_velocity = self.state[8]
        
        # Costs
        ctrl_cost = 0.1 * np.sum(np.square(action))
        
        # Rewards
        forward_reward = x_velocity

        reward = forward_reward - ctrl_cost
        
        # --- Check for termination ---
        self.steps += 1
        terminated = self.unsafe() # Always False for Cheetah
        truncated = self.steps >= self._max_episode_steps

        return self.state, reward, terminated, truncated, {}

    def seed(self, seed: int):
        super().seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rng = np.random.default_rng(seed)

    def unsafe(self) -> bool:
        """
        Determines if the current state is unsafe. For Cheetah-v4,
        there are no termination conditions besides the time limit,
        so no state is considered "unsafe".
        """
        return not -2.8795 <= self.state[9] <= 2.8795

    def simulate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Performs a one-step simulation from a given state and action
        without altering the internal state of the environment.
        """
        next_state = simulated_env_util.eval_model(self.model, state, action, stds=self.stds)
        return copy.deepcopy(next_state)


class Gymnasium_CheetahSimulate:
    """
    Wrapper class to register the simulated Cheetah environment.
    """
    environment_name = "gymnasium_cheetah_simulate"
    entry_point = "environments.gymnasium_cheetah_simulate:Gymnasium_CheetahSimulateEnv"
    max_episode_steps = 1000
    reward_threshold = 4800 # Same as the real environment

    version = 1

    def __init__(self, **kwargs):
        config = {
            'learned_model': kwargs.pop('learned_model', []),
            'stds': kwargs.pop('stds', []),
        }

        env_name = "Marvel%s-v%u" % (self.environment_name, self.version)
        self.env_name = env_name
        
        gym.register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config,
        )
        Gymnasium_CheetahSimulate.version += 1
        self._config = config
        self.__dict__.update(config)
        
        self.gym_env = gym.make(env_name, learned_model=config['learned_model'], stds=config['stds'])
        self.state = None

        # --- Plotting Functions (same as real env) ---
        def plot_other_components():
            plt.axhline(0, color='grey', linestyle='--')
            plt.grid(True, alpha=0.3)

        def plot_state_to_xy(state):
            # We use x-velocity as a proxy for x-position and z-position for y-position.
            if state is None or len(state) < 9:
                return 0, 0
            x_vel = state[8]
            z_pos = state[0]
            return x_vel, z_pos
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
