# environments/gymnasium_ant.py

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class GymnasiumAntEnv(gym.Env):
    """
    This class wraps the standard Gymnasium "Ant-v5" environment to fit
    the specific structure required by the VELM repository template.
    """
    def __init__(self):
        super(GymnasiumAntEnv, self).__init__()

        # Initialize the underlying Gymnasium environment
        self.env = gym.make("Ant-v4", use_contact_forces=True)

        # Set the action and observation spaces from the base environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # --- Define the initial state space for Ant-v5 ---
        # The observation space for Ant-v5 has 105 dimensions by default:
        # qpos (13): z-pos, root_quat (4), joint_pos (8)
        # qvel (14): root_lin_vel (3), root_ang_vel (3), joint_vel (8)
        # cfrc_ext (78): external forces on 13 body parts (13 * 6)
        
        # We define a small box around a typical starting pose (mostly zero, standing upright).
        # The full observation space is self.observation_space.low/high
        low_init = self.observation_space.low.copy()
        high_init = self.observation_space.high.copy()

        # 1. qpos (indices 0-12)
        # z-position (index 0)
        low_init[0], high_init[0] = 0.7, 0.8
        # root quaternion (indices 1-4) near [1, 0, 0, 0]
        low_init[1:5], high_init[1:5] = [0.95, -0.05, -0.05, -0.05], [1.05, 0.05, 0.05, 0.05]
        # joint positions (indices 5-12) around 0
        low_init[5:13], high_init[5:13] = -0.2, 0.2

        # 2. qvel (indices 13-26)
        # All velocities around 0
        low_init[13:27], high_init[13:27] = -0.1, 0.1
        
        # 3. cfrc_ext (indices 27-104)
        # All external forces around 0
        low_init[27:], high_init[27:] = -0.1, 0.1

        self.init_space = gym.spaces.Box(low=low_init, high=high_init, dtype=np.float64)

        self.rng = np.random.default_rng()
        self._max_episode_steps = self.env.spec.max_episode_steps
        self.state = None
        self.steps = 0

    def reset(self, seed=None,  options: dict = None, **kwargs) -> Tuple[np.ndarray, dict]:
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
        
        # 'terminated' flag from Ant-v5 handles unsafe conditions (falling)
        # 'truncated' flag handles the max episode steps
        return self.state, reward, terminated, truncated, info

    def seed(self,  seed: int = None ):
        """
        Seeds the random number generators for the environment.
        """
        super().seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        # self.init_space.seed(seed) # gym.spaces.Box does not have a seed method
        self.rng = np.random.default_rng(seed)
        # Also seed the underlying environment
        self.env.reset(seed=seed)

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
    

class GymnasiumAnt:
    """
    Wrapper class to register the Ant environment with Gymnasium according
    to the VELM repository's structure.
    """
    environment_name = "gymnasium_ant"
    entry_point = "environments.gymnasium_ant:GymnasiumAntEnv"
    max_episode_steps = 1000
    reward_threshold = 6000  # Standard reward threshold for Ant

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
        GymnasiumAnt.version += 1
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
            # For Ant-v5, the default observation does not include absolute x,y position.
            # We use the z-position for the y-axis and the torso's x-velocity as a proxy for movement on the x-axis.
            # - state[0] is the z-position of the torso.
            # - state[13] is the x-velocity of the torso.
            if len(state) < 14:
                return 0, 0
            
            x_vel = state[13] 
            z_pos = state[0]  
            
            return x_vel, z_pos
        
        self.plot_other_components = plot_other_components
        self.plot_state_to_xy = plot_state_to_xy