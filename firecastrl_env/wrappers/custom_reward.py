import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper
from typing import Callable, Dict, Any, Optional
from firecastrl_env.envs.environment.enums import FireState


def default_reward_function(
    env: gym.Env,
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any]
) -> float:
    """
    Default reward function for wildfire suppression.
    
    This function provides a balanced reward signal that:
    - Heavily penalizes newly burnt cells
    - Rewards extinguishing fires via helitack
    - Rewards reduction in burning cells
    - Penalizes ongoing fire spread (proportional to burning cells)
    - Applies a small step penalty to encourage efficiency
    
    Args:
        env: The environment instance
        prev_state: State snapshot from previous step
        curr_state: State snapshot from current step
    
    Returns:
        Reward value clipped to [-10.0, 10.0]
    """
    newly_burnt = max(0, curr_state['cells_burnt'] - prev_state['cells_burnt'])
    burning_reduction = max(0, prev_state['cells_burning'] - curr_state['cells_burning'])
    extinguished = float(curr_state.get('quenched_cells', 0))
    
    reward = 0.0
    reward -= 1.0 * newly_burnt
    reward += 2.0 * extinguished
    reward += 1.0 * burning_reduction
    reward -= 0.05 * curr_state['cells_burning']
    reward -= 0.01
    
    return float(np.clip(reward, -10.0, 10.0))


class CustomRewardWrapper(RewardWrapper):
    """
    Reward wrapper that allows users to define custom reward functions.
    
    This wrapper is designed for experimentation with different reward shaping strategies.
    It automatically captures environment state snapshots before and after each step,
    then passes them to a user-defined reward function.
    
    The state snapshot includes:
        - cells_burning: Number of currently burning cells
        - cells_burnt: Number of cells that have finished burning
        - helicopter_coord: Current helicopter position [x, y]
        - quenched_cells: Number of cells extinguished in the last step
    
    Args:
        env: The environment to wrap
        reward_fn: Custom reward function with signature:
            (env, prev_state, curr_state) -> float
            If None, uses the default reward function.
    
    Example:
        >>> import gymnasium as gym
        >>> from firecastrl_env.wrappers import CustomRewardWrapper
        >>> 
        >>> def my_reward(env, prev, curr):
        ...     # Simple example: only reward extinguishing fires
        ...     return 10.0 * curr['quenched_cells']
        >>> 
        >>> env = gym.make("firecastrl_env/WildfireEnv-v0")
        >>> env = CustomRewardWrapper(env, reward_fn=my_reward)
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Optional[Callable[[gym.Env, Dict[str, Any], Dict[str, Any]], float]] = None
    ):
        super().__init__(env)
        self.reward_fn = reward_fn if reward_fn is not None else default_reward_function
        self._prev_state: Optional[Dict[str, Any]] = None

    def reset(self, **kwargs) -> tuple:
        """Reset environment and capture initial state."""
        obs, info = self.env.reset(**kwargs)
        self._prev_state = self._capture_state()
        return obs, info

    def step(self, action: int) -> tuple:
        """Execute step and compute custom reward."""
        self._prev_state = self._capture_state()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        curr_state = self._capture_state()
        custom_reward = self.reward_fn(self.env, self._prev_state, curr_state)
        
        return obs, float(custom_reward), terminated, truncated, info

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current environment state for reward computation."""
        base_env = self.env.unwrapped
        cells = getattr(base_env, 'cells', [])
        
        cells_burning = 0
        cells_burnt = 0
        for cell in cells:
            fire_state = getattr(cell, 'fireState', FireState.Unburnt)
            if fire_state == FireState.Burning:
                cells_burning += 1
            elif fire_state == FireState.Burnt:
                cells_burnt += 1
        
        state_dict = getattr(base_env, 'state', {})
        helicopter_coord = state_dict.get('helicopter_coord', np.array([0, 0]))
        quenched_cells = state_dict.get('quenched_cells', 0)
        
        return {
            'cells_burning': cells_burning,
            'cells_burnt': cells_burnt,
            'helicopter_coord': helicopter_coord,
            'quenched_cells': float(quenched_cells if quenched_cells is not None else 0)
        }


