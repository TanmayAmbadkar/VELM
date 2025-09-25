# environments/gymnasium_hopper_simulate.py

import copy
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# This utility would contain the function to run inference with a learned dynamics model.
# Since we don't have the actual file, we'll assume it has a function:
# eval_model(model, state, action, stds) -> next_state
from . import simulated_env_util


class Gymnasium_HopperSimulateEnv(gym.Env):
    """
    This class provides a simulated version of the Hopper environment.
    It uses a learned dynamics model to predict the next state instead of
    relying on the underlying physics engine.
    """
    def __init__(self, learned_model: List[Any] = [], stds: List[float] = []):
        super(Gymnasium_HopperSimulateEnv, self).__init__()

        # --- Define spaces consistent with the real Hopper environment ---
        # Action space: 3-dimensional vector for joint torques
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation space: 11-dimensional state vector
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # Initial state space (same as the real environment)
        low_init = np.array([
            -0.05, 1.2, -0.05, -0.05, -0.05, -0.05,
            -0.05, -0.05, -0.05, -0.05, -0.05
        ])
        high_init = np.array([
            0.05, 1.3, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05
        ])
        self.init_space = gym.spaces.Box(low=low_init, high=high_init, dtype=np.float32)

        self.rng = np.random.default_rng()

        # Store the learned model and standard deviations for normalization
        self.model = copy.deepcopy(learned_model)
        self.stds = copy.deepcopy(stds)

        self._max_episode_steps = 1000
        self.state = None
        self.steps = 0
        
        # Timestep (dt) from the original Hopper environment
        self.dt = 0.008

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
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

        # Store pre-update state info needed for reward calculation
        x_position_before = self.state[0]

        # Predict the next state using the learned dynamics model
        self.state = simulated_env_util.eval_model(self.model, self.state, action, stds=self.stds)

        # --- Replicate Hopper's reward function ---
        x_position_after = self.state[0]
        forward_velocity = (x_position_after - x_position_before) / self.dt
        
        # Costs
        control_cost = 1e-3 * np.sum(np.square(action))
        
        # Rewards
        healthy_reward = 1.0
        forward_reward = forward_velocity

        reward = forward_reward - control_cost + healthy_reward
        
        # --- Check for termination ---
        self.steps += 1
        terminated = self.unsafe()
        truncated = self.steps >= self._max_episode_steps

        return self.state, reward, terminated, truncated, {}

    
    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(np.random.PCG64(seed))

    def unsafe(self) -> bool:
        """
        Checks if the *predicted* state is unsafe (i.e., the hopper has fallen).
        This uses the same conditions as the real environment.
        """
        if self.state is None:
            return False
            
        # Your custom unsafe condition from the previous file
        is_healthy = -0.37315 <= self.state[5] <= 0.37315
        return not is_healthy

    def simulate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Performs a one-step simulation from a given state and action
        without altering the internal state of the environment.
        """
        next_state = simulated_env_util.eval_model(self.model, state, action, stds=self.stds)
        return copy.deepcopy(next_state)


class Gymnasium_HopperSimulate:
    """
    Wrapper class to register the simulated Hopper environment.
    """
    environment_name = "gymnasium_hopper_simulate"
    entry_point = "environments.gymnasium_hopper_simulate:Gymnasium_HopperSimulateEnv"
    max_episode_steps = 1000
    reward_threshold = 3800 # Same as the real environment

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
        Gymnasium_HopperSimulate.version += 1
        self._config = config
        self.__dict__.update(config)
        
        # When creating the env, pass the learned model to it
        self.gym_env = gym.make(env_name, learned_model=config['learned_model'], stds=config['stds'])
        self.state = None

        # --- Plotting Functions (same as real env) ---
        def plot_other_components():
            plt.axhline(0, color='grey', linestyle='--')

        def plot_state_to_xy(state):
            x_pos = state[0]
            y_pos = state[1] # Height
            return x_pos, y_pos
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
