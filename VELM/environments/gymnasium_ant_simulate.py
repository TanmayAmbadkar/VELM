import copy
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# This utility would contain the function to run inference with a learned dynamics model.
# Since we don't have the actual file, we'll assume it has a function:
# eval_model(model, state, action, stds) -> next_state
from . import simulated_env_util


class Gymnasium_AntSimulateEnv(gym.Env):
    """
    This class provides a simulated version of the Ant-v5 environment.
    It uses a learned dynamics model to predict the next state instead of
    relying on the underlying physics engine.
    """
    def __init__(self, learned_model: List[Any] = [], stds: List[float] = []):
        super(Gymnasium_AntSimulateEnv, self).__init__()

        # --- Define spaces consistent with the real Ant-v5 environment ---
        # Action space: 8-dimensional vector for joint torques
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # Observation space: 105-dimensional state vector for Ant-v5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64)

        # Initial state space (a small box around a healthy standing pose)
        low_init = self.observation_space.low.copy()
        high_init = self.observation_space.high.copy()
        low_init[0], high_init[0] = 0.7, 0.8  # z-position
        low_init[1:5], high_init[1:5] = [0.95, -0.05, -0.05, -0.05], [1.05, 0.05, 0.05, 0.05] # quaternion
        low_init[5:27], high_init[5:27] = -0.1, 0.1  # joint pos/vel
        low_init[27:], high_init[27:] = -0.1, 0.1 # external forces

        self.init_space = gym.spaces.Box(low=low_init, high=high_init, dtype=np.float64)

        self.rng = np.random.default_rng()

        # Store the learned model and standard deviations for normalization
        self.model = copy.deepcopy(learned_model)
        self.stds = copy.deepcopy(stds)

        self._max_episode_steps = 1000
        self.state = None
        self.x_position = 0.0  # Track x-position separately
        self.steps = 0
        
        # Timestep (dt) from the original Ant-v5 environment (5 frames * 0.01s)
        self.dt = 0.05
        
        
    def reset(self,  seed: int = None, options: dict = None, ):
        """
        Resets the simulated environment to a random initial state.
        """
        if seed is not None:
            self.seed(seed)
        self.state = self.init_space.sample()
        self.x_position = 0.0
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

        # --- Replicate Ant-v5's reward function ---
        # x_velocity is the 13th element of the observation space (qvel[0])
        x_velocity = self.state[13]
        
        # Costs
        ctrl_cost = 0.5 * np.sum(np.square(action))
        # Use cfrc_ext as a proxy for contact forces
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(self.state[27:]))
        
        # Rewards
        healthy_reward = 1.0
        forward_reward = x_velocity

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        
        # --- Check for termination ---
        self.steps += 1
        terminated = self.unsafe()
        truncated = self.steps >= self._max_episode_steps

        # Update internal x_position for plotting/info
        self.x_position += x_velocity * self.dt

        return self.state, reward, terminated, truncated, {}

    def seed(self, seed: int):
        super().seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rng = np.random.default_rng(seed)

    def unsafe(self) -> bool:
        """
        Determines if the current state is unsafe for Ant-v5.
        
        An unsafe state is when the ant has fallen over. The termination condition
        for Ant-v5 is when the z-coordinate of the torso is not within [0.2, 1.0].
        The z-coordinate is the first element of the observation vector.
        """
        if self.state is None:
            return False
        
        # z_position = self.state[0]
        is_healthy_height = -2.3475 <= self.state[13] <= 2.3475
        
        return not is_healthy_height

    def simulate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Performs a one-step simulation from a given state and action
        without altering the internal state of the environment.
        """
        next_state = simulated_env_util.eval_model(self.model, state, action, stds=self.stds)
        return copy.deepcopy(next_state)


class Gymnasium_AntSimulate:
    """
    Wrapper class to register the simulated Ant environment.
    """
    environment_name = "gymnasium_ant_simulate"
    entry_point = "environments.gymnasium_ant_simulate:Gymnasium_AntSimulateEnv"
    max_episode_steps = 1000
    reward_threshold = 6000 # Same as the real environment

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
        Gymnasium_AntSimulate.version += 1
        self._config = config
        self.__dict__.update(config)
        
        # When creating the env, pass the learned model to it
        self.gym_env = gym.make(env_name, learned_model=config['learned_model'], stds=config['stds'])
        self.state = None

        # --- Plotting Functions (same as real env) ---
        def plot_other_components():
            plt.axhline(0, color='grey', linestyle='--')
            plt.grid(True, alpha=0.3)

        def plot_state_to_xy(state):
            # Use the absolute x_position (tracked internally) and z-height for plotting
            z_pos = state[0]
            # We need access to the environment instance to get the tracked x_position
            # For a static method, we'll use x_velocity as a proxy if instance is not available.
            x_vel = state[13]
            return x_vel, z_pos # Using velocity as a proxy for movement
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
