import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
from typing import Dict, Any, List, Optional

from firecastrl_env.envs.environment.enums import FireState, DroughtLevel, VegetationType


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
        
        # Get grid dimensions from base environment
        base_env = self.env.unwrapped
        self.grid_height = getattr(base_env, 'gridHeight')
        self.grid_width = getattr(base_env, 'gridWidth')
        zones = getattr(base_env, 'zones', None)
        zones_count = len(zones) if zones is not None else 1
        self._zones_norm = max(1, zones_count - 1)
        
        # Calculate feature count (position adds 2 channels)
        self.feature_count = len(self.properties) + (1 if 'position' in self.properties else 0)
        
        # Build new observation space
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

    def _normalize_value(self, value: float, cap: float) -> float:
        """Normalize a value to [0, 1] range with capping."""
        if value is None:
            return 0.0
        if np.isinf(value) or np.isnan(value):
            value = cap
        return float(np.clip(value, 0.0, cap) / cap)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform observation by adding detailed cell features."""
        base_env = self.env.unwrapped
        cells = getattr(base_env, 'cells', [])
        
        # Initialize feature tensor
        features = np.zeros((self.feature_count, self.grid_height, self.grid_width), dtype=np.float32)
        
        # Extract features based on selected properties
        for row in range(self.grid_height):
            row_start = row * self.grid_width
            for col in range(self.grid_width):
                cell_idx = row_start + col
                cell = cells[cell_idx]
                
                feature_idx = 0
                
                for prop in self.properties:
                    if prop == 'ignition_time':
                        ignition = getattr(cell, 'ignitionTime', np.inf)
                        features[feature_idx, row, col] = self._normalize_value(ignition, self.ignition_cap)
                        feature_idx += 1
                    
                    elif prop == 'spread_rate':
                        spread = getattr(cell, 'spreadRate', 0.0)
                        features[feature_idx, row, col] = self._normalize_value(spread, self.spread_cap)
                        feature_idx += 1
                    
                    elif prop == 'burn_time':
                        burn_time = getattr(cell, 'burnTime', 0.0)
                        features[feature_idx, row, col] = self._normalize_value(burn_time, self.burn_time_cap)
                        feature_idx += 1
                    
                    elif prop == 'fire_state':
                        fire_state = getattr(cell, 'fireState', FireState.Unburnt)
                        features[feature_idx, row, col] = float(fire_state) / 2.0
                        feature_idx += 1
                    
                    elif prop == 'is_river':
                        is_river = getattr(cell, 'isRiver', False)
                        features[feature_idx, row, col] = 1.0 if is_river else 0.0
                        feature_idx += 1
                    
                    elif prop == 'is_unburnt_island':
                        is_island = getattr(cell, 'isUnburntIsland', False)
                        features[feature_idx, row, col] = 1.0 if is_island else 0.0
                        feature_idx += 1
                    
                    elif prop == 'helitack_drops':
                        drops = getattr(cell, 'helitackDropCount', 0.0)
                        features[feature_idx, row, col] = self._normalize_value(drops, self.helitack_cap)
                        feature_idx += 1
                    
                    elif prop == 'elevation':
                        elevation = getattr(cell, 'baseElevation', 0.0)
                        features[feature_idx, row, col] = self._normalize_value(elevation, self.elevation_cap)
                        feature_idx += 1
                    
                    elif prop == 'zone':
                        zone_idx = int(getattr(cell, 'zoneIdx', 0) or 0)
                        features[feature_idx, row, col] = float(zone_idx) / float(self._zones_norm)
                        feature_idx += 1
                    
                    elif prop == 'vegetation':
                        zone = getattr(cell, 'zone', None)
                        veg = getattr(zone, 'vegetation', VegetationType.Barren) if zone else VegetationType.Barren
                        veg_val = int(getattr(veg, 'value', 0))
                        features[feature_idx, row, col] = float(veg_val) / 17.0
                        feature_idx += 1
                    
                    elif prop == 'drought':
                        zone = getattr(cell, 'zone', None)
                        drought = getattr(zone, 'droughtLevel', DroughtLevel.NoDrought) if zone else DroughtLevel.NoDrought
                        drought_val = int(getattr(drought, 'value', 0))
                        features[feature_idx, row, col] = float(drought_val) / 3.0
                        feature_idx += 1
                    
                    elif prop == 'position':
                        x = getattr(cell, 'x', col)
                        y = getattr(cell, 'y', row)
                        features[feature_idx, row, col] = float(x) / float(self.grid_width - 1)
                        features[feature_idx + 1, row, col] = float(y) / float(self.grid_height - 1)
                        feature_idx += 2
        
        # Create new observation dict
        new_obs = dict(obs)
        new_obs['detailed_cells'] = features
        
        # Remove basic cells if requested
        if self.remove_basic_cells and 'cells' in new_obs:
            del new_obs['cells']
        
        return new_obs


