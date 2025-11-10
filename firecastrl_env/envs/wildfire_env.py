import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Optional, Tuple, Dict, Any

from . import config
from .environment.cell import Cell
from .fire_engine.fire_engine import FireEngine
from .environment.vector import Vector2
from .environment.wind import Wind
from .environment.zone import Zone
from .environment.enums import FireState
from .environment import helper as helper


class WildfireEnv(gym.Env):
    """
    Wildfire suppression environment with heliatack-based fire control.
    
    Observation Space:
        Dict with keys:
        - 'cells': Box(160, 240) of ignition times (float32, with inf for unscheduled)
        - 'helicopter_coord': Box(2,) helicopter position [x, y] (int32)
        - 'quenched_cells': Box(1,) count of cells extinguished this step (float32)
    
    Action Space:
        Discrete(5):
        - 0: Move down
        - 1: Move up
        - 2: Move left
        - 3: Move right
        - 4: Perform helitack (extinguish fire in radius)
    
    Reward:
        Shaped reward encouraging fire suppression and discouraging spread.
    
    Args:
        env_id: Environment instance ID for loading different terrain/landcover data.
        render_mode: Optional rendering mode ('human' for matplotlib visualization).
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(self, env_id: int = 0, render_mode: Optional[str] = None):
        super().__init__()
        
        # Initialize spaces
        self.env_id = env_id
        self.render_mode = render_mode
        self._renderer = None
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'cells': spaces.Box(low=0.0, high=math.inf, shape=(160, 240), dtype=np.float32),
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'quenched_cells': spaces.Box(low=0.0, high=38400.0, shape=(1,), dtype=np.float32)
        })
        
        # Environment configuration
        self.cell_size = config.cellSize
        self.gridWidth = config.gridWidth
        self.gridHeight = config.gridHeight
        self.zones = [Zone(**zone_dict) for zone_dict in config.ZONES]
        
        self.simulation_time = 0.0
        ratio = 86400 / getattr(config, 'modelDayInSeconds', 8)
        optimal_time_step = ratio * 0.000277
        self._step_delta = min(
            getattr(config, 'maxTimeStep', 180),
            optimal_time_step * 4
        )  
        
        self._reset_state_variables()
    
    def _reset_state_variables(self) -> None:
        """Reset all episode state variables."""
        self.step_count = 0
        self.simulation_time = 0.0
        self.simulation_running = True
        self.cells = []
        self.engine = None
        self.helitack_history = []  
        self.state = {
            'cells': np.zeros((160, 240), dtype=np.float64),
            'helicopter_coord': np.array([70, 30], dtype=np.int32),
            'quenched_cells': np.array([0], dtype=np.float32),
            'last_action': None
        }
    
    def tick(self, time_step: float) -> None:
        """Advance fire simulation by given time step."""
        if self.engine and self.simulation_running:
            self.simulation_time += time_step
            self.engine.update_fire(self.cells, self.simulation_time)
            
            if self.engine.fire_did_stop:
                self.simulation_running = False

    def _get_current_fire_stats(self) -> Tuple[int, int]:
        """Return (cells_burning, cells_burnt) counts."""
        cells_burning = sum(1 for cell in self.cells if cell.fireState == FireState.Burning)
        cells_burnt = sum(1 for cell in self.cells if cell.fireState == FireState.Burnt)
        return cells_burning, cells_burnt
    
    def apply_action(self, action: int) -> Tuple[int, int, int]:
        """Apply action, return (heli_x, heli_y, quenched_cells)."""
        self.step_count += 1
        self.state['last_action'] = action
        heli_x, heli_y = self.state['helicopter_coord']
        quenched_cells = 0

        # Move helicopter
        if action == 0:   # Down
            heli_y += config.HELICOPTER_SPEED
        elif action == 1: # Up
            heli_y -= config.HELICOPTER_SPEED
        elif action == 2: # Left
            heli_x -= config.HELICOPTER_SPEED
        elif action == 3: # Right
            heli_x += config.HELICOPTER_SPEED
        elif action == 4: # Helitack
            quenched_cells = helper.perform_helitack(self.cells, heli_x, heli_y)
            self.helitack_history.append((heli_x, heli_y, self.step_count))
            # Keep only recent helitack actions (last 20 steps)
            self.helitack_history = [(x, y, s) for x, y, s in self.helitack_history if self.step_count - s < 20]
        
        # Clip coordinates to valid range
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        
        return heli_x, heli_y, quenched_cells
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset all state variables
        self._reset_state_variables()
        self.episode_count = getattr(self, 'episode_count', 0) + 1
        
        # Initialize cells
        self.cells = helper.populateCellsData(self.env_id, self.zones)
        
        # Initialize fire spark
        spark = Vector2(60000 - 1, 40000 - 1)
        grid_x = int(spark.x // self.cell_size)
        grid_y = int(spark.y // self.cell_size)
        spark_cell = self.cells[helper.get_grid_index_for_location(grid_x, grid_y, self.gridWidth)]
        spark_cell.ignitionTime = 0
        
        # Initialize fire engine
        self.engine = FireEngine(Wind(0.0, 0.0), config)
        
        observation = {
            'cells': helper.ignition_times(self.cells, self.gridWidth, self.gridHeight),
            'helicopter_coord': self.state['helicopter_coord'].copy(),
            'quenched_cells': np.array([0], dtype=np.float32)
        }
        
        return observation, {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        heli_x, heli_y, quenched_cells = self.apply_action(action)
        self.tick(self._step_delta)
        
        # Update helicopter position and fire status
        self.state['helicopter_coord'] = np.array([heli_x, heli_y], dtype=np.int32)
        self.state['quenched_cells'] = quenched_cells
        cells_burning, cells_burnt = self._get_current_fire_stats()

        # Calculate reward
        reward = self.calculate_reward(cells_burning, quenched_cells)
        
        # Check if episode is done or truncated
        terminated = (cells_burning == 0)
        truncated = (self.step_count >= config.MAX_TIMESTEPS)
        
        observation = {
            'cells': helper.ignition_times(self.cells, self.gridWidth, self.gridHeight),
            'helicopter_coord': self.state['helicopter_coord'].copy(),
            'quenched_cells': np.array([quenched_cells], dtype=np.float32)
        }
        
        info = {
            'cells_burning': cells_burning,
            'cells_burnt': cells_burnt,
            'simulation_time': self.simulation_time,
        }
        
        return observation, float(reward), terminated, truncated, info

    def calculate_reward(self, curr_burning: int, extinguished_by_helitack: int) -> float:
        """Calculate step reward based on fire state and actions."""
        reward = 0.0

        # Reward extinguishing burning cells
        reward += 5.0 * extinguished_by_helitack

        # Penalty for ongoing burning cells
        reward -= 0.05 * curr_burning

        # Small step penalty to encourage faster containment
        reward -= 0.01

        # Penalty for acting on unburnable/burnt cells
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3

        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]
            if cell_value == -1 or fire_states[heli_y, heli_x] == FireState.Burnt:
                reward -= 1.0

        return float(np.clip(reward, -10.0, 10.0))
    
    def render(self):
        """Render the environment using matplotlib (if render_mode='human')."""
        if self.render_mode is None:
            return None
            
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render to pygame window."""
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")
        
        if self._renderer is None:
            pygame.init()
            self.window_width = 960
            self.window_height = 640
            self._renderer = {}
            self._renderer['screen'] = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Wildfire Environment - FirecastRL")
            self._renderer['clock'] = pygame.time.Clock()
            self._renderer['font'] = pygame.font.Font(None, 28)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        arr = helper.ignition_times(self.cells, self.gridWidth, self.gridHeight)
        
        arr_normalized = np.copy(arr)
        arr_normalized[np.isinf(arr_normalized)] = np.nan
        finite_vals = arr_normalized[np.isfinite(arr_normalized)]
        
        if finite_vals.size > 0:
            vmax = float(np.percentile(finite_vals, 95))
            arr_normalized = np.nan_to_num(arr_normalized, nan=0.0, posinf=vmax, neginf=0.0)
            arr_normalized = np.clip(arr_normalized, 0, vmax) / vmax
        else:
            arr_normalized = np.zeros_like(arr_normalized)
        
        rgb_array = np.zeros((self.gridHeight, self.gridWidth, 3), dtype=np.uint8)
        
        for y in range(self.gridHeight):
            for x in range(self.gridWidth):
                val = arr_normalized[y, x]
                # Flip Y coordinate to match landcover orientation
                cell_y = self.gridHeight - 1 - y
                cell = self.cells[cell_y * self.gridWidth + x]
                
                if cell.isRiver:
                    rgb_array[y, x] = [30, 60, 120]
                elif cell.fireState == FireState.Burning:
                    rgb_array[y, x] = [255, 100, 0]
                elif cell.fireState == FireState.Burnt:
                    rgb_array[y, x] = [60, 60, 60]
                elif val > 0:
                    if val < 0.33:
                        rgb_array[y, x] = [int(val * 3 * 180), 0, int(val * 3 * 50)]
                    elif val < 0.66:
                        t = (val - 0.33) / 0.33
                        rgb_array[y, x] = [180 + int(t * 75), int(t * 50), 50]
                    else:
                        t = (val - 0.66) / 0.34
                        rgb_array[y, x] = [255, 50 + int(t * 100), 50 + int(t * 150)]
                else:
                    rgb_array[y, x] = [20, 20, 20]
        
        rgb_array_scaled = np.repeat(np.repeat(rgb_array, 4, axis=0), 4, axis=1)
        
        surface = pygame.surfarray.make_surface(np.transpose(rgb_array_scaled, (1, 0, 2)))
        self._renderer['screen'].blit(surface, (0, 0))
        
        # Draw helitack locations 
        for hx_attack, hy_attack, attack_step in self.helitack_history:
            age = self.step_count - attack_step
            alpha = max(0.2, 1.0 - age / 20.0)  # Fade over time
            attack_screen_x = hx_attack * 4 + 2
            attack_screen_y = hy_attack * 4 + 2
            
            blue_intensity = int(alpha * 100)
            pygame.draw.circle(self._renderer['screen'], (0, 0, blue_intensity), 
                              (attack_screen_x, attack_screen_y), 8) 
        
        # Draw helicopter (yellow X with center and rotors)
        hx, hy = int(self.state['helicopter_coord'][0]), int(self.state['helicopter_coord'][1])
        heli_screen_x = hx * 4 + 2
        heli_screen_y = hy * 4 + 2
        
        # Helicopter colors
        heli_color = (255, 255, 0)  # Yellow
        heli_outline = (0, 0, 0)     # Black outline
        
        # Main cross (X shape)
        pygame.draw.line(self._renderer['screen'], heli_outline, 
                        (heli_screen_x - 9, heli_screen_y - 9), 
                        (heli_screen_x + 9, heli_screen_y + 9), 4)
        pygame.draw.line(self._renderer['screen'], heli_outline, 
                        (heli_screen_x + 9, heli_screen_y - 9), 
                        (heli_screen_x - 9, heli_screen_y + 9), 4)
        pygame.draw.line(self._renderer['screen'], heli_color, 
                        (heli_screen_x - 8, heli_screen_y - 8), 
                        (heli_screen_x + 8, heli_screen_y + 8), 2)
        pygame.draw.line(self._renderer['screen'], heli_color, 
                        (heli_screen_x + 8, heli_screen_y - 8), 
                        (heli_screen_x - 8, heli_screen_y + 8), 2)
        
        # Center body
        pygame.draw.circle(self._renderer['screen'], heli_outline, 
                          (heli_screen_x, heli_screen_y), 5)
        pygame.draw.circle(self._renderer['screen'], heli_color, 
                          (heli_screen_x, heli_screen_y), 3)
        
        # Rotor blades (horizontal line)
        pygame.draw.line(self._renderer['screen'], heli_outline, 
                        (heli_screen_x - 12, heli_screen_y), 
                        (heli_screen_x + 12, heli_screen_y), 3)
        pygame.draw.line(self._renderer['screen'], (200, 200, 200), 
                        (heli_screen_x - 11, heli_screen_y), 
                        (heli_screen_x + 11, heli_screen_y), 1)
        
        cells_burning, cells_burnt = self._get_current_fire_stats()
        text = self._renderer['font'].render(
            f"Step: {self.step_count} | Burning: {cells_burning} | Burnt: {cells_burnt}",
            True,
            (255, 255, 255),
            (0, 0, 0)
        )
        self._renderer['screen'].blit(text, (10, 10))
        
        pygame.display.flip()
        self._renderer['clock'].tick(self.metadata["render_fps"])
        
        return None
    
    def _render_rgb_array(self):
        """Render to RGB array (for video recording)."""
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")
        
        if not pygame.get_init():
            pygame.init()
        
        arr = helper.ignition_times(self.cells, self.gridWidth, self.gridHeight)
        
        arr_normalized = np.copy(arr)
        arr_normalized[np.isinf(arr_normalized)] = np.nan
        finite_vals = arr_normalized[np.isfinite(arr_normalized)]
        
        if finite_vals.size > 0:
            vmax = float(np.percentile(finite_vals, 95))
            arr_normalized = np.nan_to_num(arr_normalized, nan=0.0, posinf=vmax, neginf=0.0)
            arr_normalized = np.clip(arr_normalized, 0, vmax) / vmax
        else:
            arr_normalized = np.zeros_like(arr_normalized)
        
        rgb_array = np.zeros((self.gridHeight, self.gridWidth, 3), dtype=np.uint8)
        
        for y in range(self.gridHeight):
            for x in range(self.gridWidth):
                val = arr_normalized[y, x]
                # Flip Y coordinate to match landcover orientation
                cell_y = self.gridHeight - 1 - y
                cell = self.cells[cell_y * self.gridWidth + x]
                
                if cell.isRiver:
                    rgb_array[y, x] = [30, 60, 120]
                elif cell.fireState == FireState.Burning:
                    rgb_array[y, x] = [255, 100, 0]
                elif cell.fireState == FireState.Burnt:
                    rgb_array[y, x] = [60, 60, 60]
                elif val > 0:
                    if val < 0.33:
                        rgb_array[y, x] = [int(val * 3 * 180), 0, int(val * 3 * 50)]
                    elif val < 0.66:
                        t = (val - 0.33) / 0.33
                        rgb_array[y, x] = [180 + int(t * 75), int(t * 50), 50]
                    else:
                        t = (val - 0.66) / 0.34
                        rgb_array[y, x] = [255, 50 + int(t * 100), 50 + int(t * 150)]
                else:
                    rgb_array[y, x] = [20, 20, 20]
        
        rgb_array_scaled = np.repeat(np.repeat(rgb_array, 4, axis=0), 4, axis=1)
        
        # Draw helitack locations (filled dark blue circles)
        for hx_attack, hy_attack, attack_step in self.helitack_history:
            age = self.step_count - attack_step
            alpha_factor = max(0.2, 1.0 - age / 20.0)  # Fade over time
            attack_center_x = hx_attack * 4 + 2
            attack_center_y = hy_attack * 4 + 2
            
            # Draw filled dark blue circle
            blue_intensity = int(100 * alpha_factor)
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx*dx + dy*dy <= 64:  # Circle radius ~8, filled
                        px, py = attack_center_x + dx, attack_center_y + dy
                        if 0 <= px < rgb_array_scaled.shape[1] and 0 <= py < rgb_array_scaled.shape[0]:
                            rgb_array_scaled[py, px] = [0, 0, blue_intensity]
        
        # Draw helicopter (yellow X with center and rotors)
        hx, hy = int(self.state['helicopter_coord'][0]), int(self.state['helicopter_coord'][1])
        heli_center_x = hx * 4 + 2
        heli_center_y = hy * 4 + 2
        
        yellow = [255, 255, 0]
        black = [0, 0, 0]
        
        # X shape with outline
        for offset in range(-8, 9):
            # Diagonal lines
            x1, y1 = heli_center_x + offset, heli_center_y + offset
            x2, y2 = heli_center_x + offset, heli_center_y - offset
            
            # First diagonal (outline)
            for thick in [-1, 0, 1]:
                if 0 <= y1 < rgb_array_scaled.shape[0] and 0 <= x1 < rgb_array_scaled.shape[1]:
                    if 0 <= x1 + thick < rgb_array_scaled.shape[1]:
                        rgb_array_scaled[y1, x1 + thick] = black if abs(offset) > 6 else yellow
                
                # Second diagonal (outline)
                if 0 <= y2 < rgb_array_scaled.shape[0] and 0 <= x2 < rgb_array_scaled.shape[1]:
                    if 0 <= x2 + thick < rgb_array_scaled.shape[1]:
                        rgb_array_scaled[y2, x2 + thick] = black if abs(offset) > 6 else yellow
        
        # Center circle
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx*dx + dy*dy <= 9:
                    px, py = heli_center_x + dx, heli_center_y + dy
                    if 0 <= px < rgb_array_scaled.shape[1] and 0 <= py < rgb_array_scaled.shape[0]:
                        rgb_array_scaled[py, px] = yellow
        
        # Rotor line (horizontal)
        for rx in range(-11, 12):
            px, py = heli_center_x + rx, heli_center_y
            if 0 <= px < rgb_array_scaled.shape[1] and 0 <= py < rgb_array_scaled.shape[0]:
                rgb_array_scaled[py, px] = [200, 200, 200]
        
        return rgb_array_scaled

    def close(self):
        """Clean up environment resources."""
        if self._renderer is not None:
            try:
                import pygame
                if pygame.get_init():
                    pygame.quit()
            except Exception:
                pass
            self._renderer = None
            
        self.simulation_running = False
        self.cells = []
        self.engine = None
        self.state = None