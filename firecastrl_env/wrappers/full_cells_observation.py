import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
from typing import Dict, Any, List, Optional


class CellObservationWrapper(ObservationWrapper):
    """
    Observation wrapper that exposes detailed cell properties as multi-channel tensors.
    
    This wrapper allows users to selectively include cell properties by passing a list
    of property names. All features are normalized to [0, 1] range for stable RL training.
    
    Available properties:
        - 'ignition_time': Time when cell ignites (inf for unscheduled)
        - 'spread_rate': Fire spread rate at cell
        - 'burn_time': Remaining burn time
        - 'fire_state': Current fire state (Unburnt=0, Burning=1, Burnt=2)
        - 'is_river': Whether cell is a river (non-burnable)
        - 'is_unburnt_island': Whether cell is a designated unburnt island
        - 'helitack_drops': Number of helitack drops on cell
        - 'elevation': Base elevation of cell
        - 'zone': Zone index for cell
        - 'vegetation': Vegetation type (enum value)
        - 'drought': Drought level (enum value)
        - 'position': Cell x,y coordinates (adds 2 channels)
    
    Args:
        env: The environment to wrap
        properties: List of property names to include. If None or 'all', includes all properties.
                   Default: ['ignition_time', 'spread_rate', 'fire_state', 'is_river', 'helitack_drops', 'elevation']
        remove_basic_cells: Remove 'cells' key from observation to avoid duplication (default: True)
        ignition_cap: Max value for ignition time normalization (default: 1e5)
        spread_cap: Max value for spread rate normalization (default: 1e3)
        burn_time_cap: Max value for burn time normalization (default: 1e4)
        elevation_cap: Max value for elevation normalization (default: 1e4)
        helitack_cap: Max value for helitack count normalization (default: 10.0)
    
    Examples:
        >>> import gymnasium as gym
        >>> from firecastrl_env.wrappers import CellObservationWrapper
        >>> 
        >>> # Use default properties
        >>> env = gym.make("firecastrl_env/WildfireEnv-v0")
        >>> env = CellObservationWrapper(env)
        >>> 
        >>> # Include only specific properties
        >>> env = CellObservationWrapper(env, properties=['ignition_time', 'fire_state', 'is_river'])
        >>> 
        >>> # Include all available properties
        >>> env = CellObservationWrapper(env, properties='all')
    """
    
    AVAILABLE_PROPERTIES = [
        'ignition_time', 'spread_rate', 'burn_time', 'fire_state',
        'is_river', 'is_unburnt_island', 'helitack_drops', 'elevation',
        'zone', 'vegetation', 'drought', 'position'
    ]
    
    DEFAULT_PROPERTIES = [
        'ignition_time', 'spread_rate', 'fire_state',
        'is_river', 'helitack_drops', 'elevation'
    ]

    def __init__(
        self,
        env: gym.Env,
        properties: Optional[List[str]] = None,
        remove_basic_cells: bool = True,
        ignition_cap: float = 1e5,
        spread_cap: float = 1e3,
        burn_time_cap: float = 1e4,
        elevation_cap: float = 1e4,
        helitack_cap: float = 10.0,
    ):
        super().__init__(env)
        
        # Handle properties selection
        if properties is None:
            self.properties = self.DEFAULT_PROPERTIES.copy()
        elif properties == 'all':
            self.properties = self.AVAILABLE_PROPERTIES.copy()
        else:
            invalid = [p for p in properties if p not in self.AVAILABLE_PROPERTIES]
            if invalid:
                raise ValueError(
                    f"Invalid properties: {invalid}. "
                    f"Available properties: {self.AVAILABLE_PROPERTIES}"
                )
            self.properties = list(properties)
        
        self.remove_basic_cells = remove_basic_cells
        
        # Normalization caps
        self.ignition_cap = float(ignition_cap)
        self.spread_cap = float(spread_cap)
        self.burn_time_cap = float(burn_time_cap)
        self.elevation_cap = float(elevation_cap)
        self.helitack_cap = float(helitack_cap)
        
        base_env = self.env.unwrapped
        self.grid_height = getattr(base_env, "gridHeight")
        self.grid_width = getattr(base_env, "gridWidth")
        
        self.feature_count = len(self.properties) + (1 if "position" in self.properties else 0)
        
        base = self.env.observation_space
        assert isinstance(base, spaces.Dict), "CellObservationWrapper expects Dict observation space"
        
        new_spaces = dict(base.spaces)
        
        # Remove basic cells if requested
        if self.remove_basic_cells and 'cells' in new_spaces:
            del new_spaces['cells']
        
        # Add detailed cells
        new_spaces['detailed_cells'] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_count, self.grid_height, self.grid_width),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        base_env = self.env.unwrapped
        features = base_env.get_detailed_cell_observation(
            self.properties,
            ignition_cap=self.ignition_cap,
            spread_cap=self.spread_cap,
            burn_time_cap=self.burn_time_cap,
            elevation_cap=self.elevation_cap,
            helitack_cap=self.helitack_cap,
        )
        
        new_obs = dict(obs)
        new_obs["detailed_cells"] = features
        
        if self.remove_basic_cells and "cells" in new_obs:
            del new_obs["cells"]
        
        return new_obs


