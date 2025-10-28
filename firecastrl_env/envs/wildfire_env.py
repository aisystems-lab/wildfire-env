from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import time
import numpy as np
from . import config
from .environment.cell import Cell
from .fire_engine.fire_engine import FireEngine
from .environment.vector import Vector2
from .environment.wind import Wind
from .environment.zone import Zone
from .environment.enums import FireState
from .environment import helper as helper
import traceback
import math

class WildfireEnv(gym.Env):
    def __init__(self, env_id=0):
        super().__init__()
        
        # Initialize spaces
        self.env_id = env_id
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
        self.zones: list[Zone] = [Zone(**zone_dict) for zone_dict in config.ZONES]
        
        self._reset_state_variables()  # Initialize state variables
    
    def _reset_state_variables(self):
        self.step_count = 0
        self.time = 0
        self.prev_tick_time = None
        self.simulation_running = True
        self.cells = []
        self.engine = None
        self.state = {
            'cells': np.zeros((160, 240), dtype=np.float64),
            'helicopter_coord': np.array([70, 30], dtype=np.int32),
            'quenched_cells': np.array([0], dtype=np.float32),
            'last_action': None
        }
    
    def tick(self, time_step: float):
        if self.engine and self.simulation_running:
            self.time += time_step
            self.engine.update_fire(self.cells, self.time)
            
            if self.engine.fire_did_stop:
                self.simulation_running = False
                print(f"[Env {self.env_id}] Fire simulation stopped at time {self.time}")

    def _get_current_fire_stats(self):
            cells_burning = len([cell for cell in self.cells if cell.fireState == FireState.Burning])
            cells_burnt = len([cell for cell in self.cells if cell.fireState == FireState.Burnt])
            return cells_burning, cells_burnt
            
    def step_simulation(self, current_time_ms: float):
        if not self.simulation_running:
            return
        if self.prev_tick_time is None:
            self.prev_tick_time = current_time_ms
            time_step = 1.0  # Default first step
        else:
            real_time_diff_minutes = (current_time_ms - self.prev_tick_time) / 60000
            self.prev_tick_time = current_time_ms
            
            ratio = 86400 / getattr(config, 'modelDayInSeconds', 8)
            optimal_time_step = ratio * 0.000277
            time_step = min(
                getattr(config, 'maxTimeStep', 180),
                optimal_time_step * 4,
                ratio * real_time_diff_minutes
            )
        self.tick(time_step)
    
    def apply_action(self, action):
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
            print(f"[Env {self.env_id}] Helitack at ({heli_x}, {heli_y})")
            if quenched_cells > 0:
                print(f"[Env {self.env_id}] Helitack quenched {quenched_cells} cells")
        
        # Clip coordinates to valid range
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        
        return heli_x, heli_y, quenched_cells
    
    def reset(self, *, seed=None, options=None):

            super().reset(seed=config.SEED)
            
            # Reset all state variables
            self._reset_state_variables()
            self.episode_count = getattr(self, 'episode_count', 0) + 1
            self.cells = helper.populateCellsData(self.env_id, self.zones) # Initialize cells
            
            # Initialize fire spark
            spark = Vector2(60000 - 1, 40000 - 1)
            grid_x = int(spark.x // self.cell_size)
            grid_y = int(spark.y // self.cell_size)
            spark_cell : Cell = self.cells[helper.get_grid_index_for_location(grid_x, grid_y, self.gridWidth)]
            spark_cell.ignitionTime = 0
            
            # Initialize fire engine
            self.engine = FireEngine(Wind(0.0, 0.0), config)
            observation = {
                'cells': helper.ignition_times(self.cells, self.gridWidth, self.gridHeight),
                'helicopter_coord': self.state['helicopter_coord'],
                'quenched_cells': np.array([0], dtype=np.float32)
            }
            
            return observation, {}
            
    
    def step(self, action):
            # Apply action and get helicopter position
            heli_x, heli_y, quenched_cells = self.apply_action(action)
            
            current_time_ms = time.time() * 1000 
            self.step_simulation(current_time_ms) # Run simulation step
            
            # Update helicopter position and fire status
            self.state['helicopter_coord'] = np.array([heli_x, heli_y], dtype=np.int32)
            self.state['quenched_cells'] = quenched_cells
            cells_burning, cells_burnt = self._get_current_fire_stats()

            # Calculate reward
            reward = self.calculate_reward(
                cells_burning,
                quenched_cells
            )
            
            # Check if episode is done or truncated
            terminated = (cells_burning == 0)
            truncated = (self.step_count >= config.MAX_TIMESTEPS)
            
            observation = {
                'cells': helper.ignition_times(self.cells, self.gridWidth, self.gridHeight),
                'helicopter_coord': self.state['helicopter_coord'],
                'quenched_cells': np.array([quenched_cells], dtype=np.float32)
            }
            
            print(f"[Env {self.env_id}] Step {self.step_count}: Burning={cells_burning}, Burnt={cells_burnt}, Reward={reward:.3f}")

            return observation, reward, terminated, truncated, {}

    def calculate_reward(self, curr_burning, extinguished_by_helitack):
        reward = 0.0

        # Reward extinguishing burning cells — encourage actively putting out fires
        reward += 5.0 * extinguished_by_helitack  # Larger reward for extinguishing

        # Small penalty for ongoing burning cells to discourage letting fires burn longer
        reward -= 0.05 * curr_burning

        # Small step penalty to encourage faster containment
        reward -= 0.01

        # Gentle penalty for acting on unburnable/burnt cells (to avoid wasting moves)
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3

        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]
            if cell_value == -1 or fire_states[heli_y, heli_x] == FireState.Burnt:
                reward -= 0.2  # Slightly stronger penalty here

        # Clip reward to a reasonable range for stability
        reward = np.clip(reward, -10, 10)
        return reward

    def close(self):
        print(f"Closing environment {self.env_id}...")
        self.simulation_running = False
        self.cells = []
        self.engine = None
        self.state = None