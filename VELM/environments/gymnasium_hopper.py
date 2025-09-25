# environments/gymnasium_hopper.py

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class GymnasiumHopperEnv(gym.Env):
    """
    This class wraps the standard Gymnasium "Hopper-v5" environment to fit
    the specific structure required by the VELM repository template.
    """
    def __init__(self):
        super(GymnasiumHopperEnv, self).__init__()

        # Initialize the underlying Gymnasium environment
        self.env = gym.make("Hopper-v4")

        # Set the action and observation spaces from the base environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Define the initial state space as a small box around the default reset state.
        # The default Hopper state has 11 dimensions.
        # [pos_x, pos_z, ang_y, ang_z, ang_x, vel_x, vel_z, vel_ang_y, vel_ang_z, vel_ang_x, vel_root_x]
        # We'll create a small box around the typical starting pose.
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
        self._max_episode_steps = self.env._max_episode_steps
        self.state = None
        self.steps = 0

    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state.
        """
        if seed is not None:
            self.seed(seed)
        
        # We use the base environment's reset logic
        self.state, info = self.env.reset(seed=seed)
        self.steps = 0
        return self.state, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        """
        Advances the environment by one timestep.
        """
        # Pass the action to the underlying environment
        self.state, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        
        # The 'terminated' flag from Hopper-v5 already handles unsafe conditions
        # The 'truncated' flag handles the max episode steps
        return self.state, reward, terminated, truncated, info

    def seed(self, seed: int):
        """
        Seeds the random number generators for the environment.
        """
        super().seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)
        self.rng = np.random.default_rng(seed)
        # Also seed the underlying environment
        self.env.reset(seed=seed)

    def unsafe(self) -> bool:
        """
        Determines if the current state is unsafe, based on Hopper's termination
        conditions (excluding timeout).
        An unsafe state is when the hopper has fallen.
        - z-coordinate of the torso is less than 0.7 meters.
        - Absolute value of the angle is greater than 0.2 radians.
        """
        if self.state is None:
            return False
        
        is_healthy = -0.37315 <= self.state[5] <= 0.37315  # ang_y
        
        return not is_healthy

class GymnasiumHopper:
    """
    Wrapper class to register the Hopper environment with Gymnasium according
    to the VELM repository's structure.
    """
    environment_name = "gymnasium_hopper"
    entry_point = "environments.gymnasium_hopper:GymnasiumHopperEnv"
    max_episode_steps = 1000
    reward_threshold = 3800 # Standard reward threshold for Hopper-v5

    version = 1

    def __init__(self, **kwargs):
        config = {} # No special config needed for this env

        env_name = "Marvel%s-v%u" % (self.environment_name, self.version)
        self.env_name = env_name
        
        gym.register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config,
        )
        GymnasiumHopper.version += 1
        self._config = config
        self.__dict__.update(config)
        self.gym_env = gym.make(env_name)
        self.state = None

        # Configuration for Lagrange-based safety algorithms
        self.lagrange_config = {
            "dso_dataset_size": 5000,
            "num_traj": 200,
            "horizon": self.max_episode_steps,
            "alpha": 0.001,
            "N_of_directions": 3,
            "b": 2,
            "noise": 0.001,
            "initial_lambda": 0.5,
            "iters_run": 1,
        }

        # --- Plotting Functions ---
        def plot_other_components():
            # Draw a horizontal line representing the ground
            plt.axhline(0, color='grey', linestyle='--')

        def plot_state_to_xy(state):
            # The hopper moves along the x-axis. The y-axis in the plot
            # will represent the hopper's height (z-position in the state).
            x_pos = state[0]
            y_pos = state[1] # This is actually the height (z-coordinate)
            return x_pos, y_pos
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
