from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class GymnasiumCheetahEnv(gym.Env):
    """
    This class wraps the standard Gymnasium "Cheetah-v4" environment to fit
    the specific structure required by the VELM repository template.
    """
    def __init__(self):
        super(GymnasiumCheetahEnv, self).__init__()

        # Initialize the underlying Gymnasium environment
        self.env = gym.make("HalfCheetah-v4")

        # Set the action and observation spaces from the base environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # --- Define the initial state space for Cheetah-v4 ---
        # The observation space for Cheetah-v4 has 17 dimensions by default:
        # qpos (8): z-pos, y-rotation, and 6 joint angles. Note: x-pos is excluded.
        # qvel (9): x-vel, z-vel, y-angular-vel, and 6 joint velocities.
        
        # We define a small box around a typical starting pose (mostly zero).
        # The base environment initializes with small random noise around the default state.
        low_init = -0.1 * np.ones(17)
        high_init = 0.1 * np.ones(17)

        self.init_space = gym.spaces.Box(low=low_init, high=high_init, dtype=np.float64)

        self.rng = np.random.default_rng()
        self._max_episode_steps = self.env.spec.max_episode_steps
        self.state = None
        self.steps = 0

    def reset(self, seed=None, options: dict = None, **kwargs) -> Tuple[np.ndarray, dict]:
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
        
        # 'terminated' is always False for Cheetah-v4
        # 'truncated' handles the max episode steps
        return self.state, reward, terminated, truncated, info

    def seed(self, seed: int = None ):
        """
        Seeds the random number generators for the environment.
        """
        super().seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.rng = np.random.default_rng(seed)
        # Also seed the underlying environment
        self.env.reset(seed=seed)

    def unsafe(self) -> bool:
        """
        Determines if the current state is unsafe. For Cheetah-v4,
        there are no termination conditions besides the time limit,
        so no state is considered "unsafe" in this context.
        """
        return not -2.8795 <= self.state[9] <= 2.8795
    
class GymnasiumCheetah:
    """
    Wrapper class to register the Cheetah environment with Gymnasium according
    to the VELM repository's structure.
    """
    environment_name = "gymnasium_Cheetah"
    entry_point = "environments.gymnasium_cheetah:GymnasiumCheetahEnv"
    max_episode_steps = 1000
    reward_threshold = 4800  # Standard reward threshold for Cheetah

    version = 1

    def __init__(self, **kwargs):
        config = {}  # No special config needed for this env

        env_name = "Marvel%s-v%u" % (self.environment_name, self.version)
        self.env_name = env_name
        
        gym.register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config,
        )
        GymnasiumCheetah.version += 1
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
            # Draw a horizontal plane representing the ground
            plt.axhline(0, color='grey', linestyle='--')
            plt.grid(True, alpha=0.3)

        def plot_state_to_xy(state):
            # For Cheetah, x and z positions can be retrieved from the underlying environment state.
            # However, the standard observation only contains x-velocity. We can integrate it.
            # For simplicity in this wrapper, we'll plot x-velocity vs. a constant height.
            # The 'info' dict from step() contains 'x_position' and 'x_velocity'.
            # A more robust implementation would track the position externally.
            if state is None or len(state) < 9:
                return 0, 0
            
            x_vel = state[8] # x-velocity
            # The observation does not contain height (z-position). We can't plot it directly.
            # We'll return velocity as x and 0 for y for visualization consistency.
            return x_vel, 0
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy
