import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config
from .environment import helper as helper
from .environment.enums import FireState
from .environment.vector import Vector2
from .environment.wind import Wind
from .environment.zone import Zone
from .fire_engine.fire_engine import FireEngine

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")


class WildfireEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, env_id: int = 0, render_mode: Optional[str] = None):
        super().__init__()

        self.env_id = env_id
        self.render_mode = render_mode
        self._renderer = None

        self.cell_size = config.cellSize
        self.gridWidth = config.gridWidth
        self.gridHeight = config.gridHeight
        self.zones = [Zone(**zone_dict) for zone_dict in config.ZONES]
        self.cell_state = helper.create_array_fire_state(self.env_id, self.zones)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "cells": spaces.Box(low=0.0, high=1e6, shape=(self.gridHeight, self.gridWidth), dtype=np.float32),
            "helicopter_coord": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([self.gridWidth - 1, self.gridHeight - 1]),
                dtype=np.int32,
            ),
            "quenched_cells": spaces.Box(low=0.0, high=float(self.gridWidth * self.gridHeight), shape=(1,), dtype=np.float32),
        })

        ratio = 86400 / getattr(config, "modelDayInSeconds", 8)
        optimal_time_step = ratio * 0.000277
        self._step_delta = min(getattr(config, "maxTimeStep", 180), optimal_time_step * 4)

        self._position_channels = self._build_position_channels()
        self._reset_state_variables()

    def _build_position_channels(self) -> np.ndarray:
        x_denom = max(1, self.gridWidth - 1)
        y_denom = max(1, self.gridHeight - 1)
        x_channel = np.broadcast_to(
            np.arange(self.gridWidth, dtype=np.float32) / float(x_denom),
            (self.gridHeight, self.gridWidth),
        )
        y_channel = np.broadcast_to(
            (np.arange(self.gridHeight, dtype=np.float32) / float(y_denom))[:, None],
            (self.gridHeight, self.gridWidth),
        )
        return np.stack((x_channel, y_channel), axis=0)

    def _reset_state_variables(self) -> None:
        self.step_count = 0
        self.simulation_time = 0.0
        self.simulation_running = True
        self.engine = None
        self.helitack_history = []
        self.cell_state.reset_dynamic()
        self.state = {
            "cells": np.zeros((self.gridHeight, self.gridWidth), dtype=np.float32),
            "helicopter_coord": np.array([70, 30], dtype=np.int32),
            "quenched_cells": np.array([0], dtype=np.float32),
            "last_action": None,
        }

    def _clip_ignition_grid(self) -> np.ndarray:
        return np.clip(self.cell_state.ignition_time, 0.0, 1e6).astype(np.float32, copy=False)

    def _normalize_grid(self, values: np.ndarray, cap: float) -> np.ndarray:
        arr = np.array(values, dtype=np.float32, copy=True)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=cap, neginf=0.0)
        np.clip(arr, 0.0, cap, out=arr)
        arr /= cap
        return arr

    def _get_current_fire_stats(self) -> Tuple[int, int]:
        cells_burning = int(np.count_nonzero(self.cell_state.fire_state == FireState.Burning))
        cells_burnt = int(np.count_nonzero(self.cell_state.fire_state == FireState.Burnt))
        return cells_burning, cells_burnt

    def tick(self, time_step: float) -> None:
        if self.engine and self.simulation_running:
            self.simulation_time += time_step
            self.engine.update_fire_array(self.cell_state, self.simulation_time)
            if self.engine.fire_did_stop:
                self.simulation_running = False

    def apply_action(self, action: int) -> Tuple[int, int, int]:
        self.step_count += 1
        self.state["last_action"] = action
        heli_x, heli_y = self.state["helicopter_coord"]
        quenched_cells = 0

        if action == 0:
            heli_y += config.HELICOPTER_SPEED
        elif action == 1:
            heli_y -= config.HELICOPTER_SPEED
        elif action == 2:
            heli_x -= config.HELICOPTER_SPEED
        elif action == 3:
            heli_x += config.HELICOPTER_SPEED
        elif action == 4:
            quenched_cells = helper.perform_helitack_array(self.cell_state, int(heli_x), int(heli_y))
            self.helitack_history.append((int(heli_x), int(heli_y), self.step_count))
            self.helitack_history = [(x, y, s) for x, y, s in self.helitack_history if self.step_count - s < 20]

        heli_x = int(np.clip(heli_x, 0, self.gridWidth - 1))
        heli_y = int(np.clip(heli_y, 0, self.gridHeight - 1))
        return heli_x, heli_y, quenched_cells

    def _build_observation(self, quenched_cells: int) -> Dict[str, np.ndarray]:
        observation = {
            "cells": self._clip_ignition_grid().copy(),
            "helicopter_coord": self.state["helicopter_coord"].copy(),
            "quenched_cells": np.array([quenched_cells], dtype=np.float32),
        }
        self.state["cells"] = observation["cells"]
        return observation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state_variables()
        self.episode_count = getattr(self, "episode_count", 0) + 1

        spark = Vector2(60000 - 1, 40000 - 1)
        grid_x = int(spark.x // self.cell_size)
        grid_y = int(spark.y // self.cell_size)
        self.cell_state.ignition_time[grid_y, grid_x] = 0.0

        self.engine = FireEngine(Wind(0.0, 0.0), config)
        observation = self._build_observation(0)
        return observation, {}

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        heli_x, heli_y, quenched_cells = self.apply_action(action)
        self.tick(self._step_delta)

        self.state["helicopter_coord"] = np.array([heli_x, heli_y], dtype=np.int32)
        self.state["quenched_cells"] = np.array([quenched_cells], dtype=np.float32)
        cells_burning, cells_burnt = self._get_current_fire_stats()

        reward = self.calculate_reward(cells_burning, quenched_cells)
        terminated = cells_burning == 0
        truncated = self.step_count >= config.MAX_TIMESTEPS

        observation = self._build_observation(quenched_cells)
        info = {
            "cells_burning": cells_burning,
            "cells_burnt": cells_burnt,
            "simulation_time": self.simulation_time,
        }
        return observation, float(reward), terminated, truncated, info

    def calculate_reward(self, curr_burning: int, extinguished_by_helitack: int) -> float:
        reward = 0.0
        reward += 5.0 * extinguished_by_helitack
        reward -= 0.05 * curr_burning
        reward -= 0.01

        heli_x, heli_y = self.state["helicopter_coord"]
        last_action = self.state.get("last_action", None)
        if last_action == 4 and 0 <= heli_y < self.gridHeight and 0 <= heli_x < self.gridWidth:
            if (
                self.cell_state.is_nonburnable[heli_y, heli_x]
                or self.cell_state.fire_state[heli_y, heli_x] == FireState.Burnt
            ):
                reward -= 1.0

        return float(np.clip(reward, -10.0, 10.0))

    def get_detailed_cell_observation(
        self,
        properties,
        *,
        ignition_cap: float = 1e5,
        spread_cap: float = 1e3,
        burn_time_cap: float = 1e4,
        elevation_cap: float = 1e4,
        helitack_cap: float = 10.0,
    ) -> np.ndarray:
        features = []
        for prop in properties:
            if prop == "ignition_time":
                features.append(self._normalize_grid(self.cell_state.ignition_time, ignition_cap))
            elif prop == "spread_rate":
                features.append(self._normalize_grid(self.cell_state.spread_rate, spread_cap))
            elif prop == "burn_time":
                features.append(self._normalize_grid(self.cell_state.burn_time, burn_time_cap))
            elif prop == "fire_state":
                features.append(self.cell_state.fire_state.astype(np.float32) / 2.0)
            elif prop == "is_river":
                features.append(self.cell_state.is_river.astype(np.float32))
            elif prop == "is_unburnt_island":
                features.append(self.cell_state.is_unburnt_island.astype(np.float32))
            elif prop == "helitack_drops":
                features.append(self._normalize_grid(self.cell_state.helitack_drops, helitack_cap))
            elif prop == "elevation":
                features.append(self._normalize_grid(self.cell_state.base_elevation, elevation_cap))
            elif prop == "zone":
                zones_norm = max(1, len(self.zones) - 1)
                features.append(self.cell_state.zone_idx.astype(np.float32) / float(zones_norm))
            elif prop == "vegetation":
                features.append(self.cell_state.vegetation.astype(np.float32) / 17.0)
            elif prop == "drought":
                features.append(self.cell_state.drought_level.astype(np.float32) / 3.0)
            elif prop == "position":
                features.extend(list(self._position_channels))

        return np.asarray(features, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            return self._render_human()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _build_render_base(self) -> np.ndarray:
        ignition = np.flipud(self.cell_state.ignition_time)
        fire_state = np.flipud(self.cell_state.fire_state)
        is_river = np.flipud(self.cell_state.is_river)

        arr_normalized = np.array(ignition, dtype=np.float32, copy=True)
        arr_normalized[np.isinf(arr_normalized)] = np.nan
        finite_vals = arr_normalized[np.isfinite(arr_normalized)]

        if finite_vals.size > 0:
            vmax = float(np.percentile(finite_vals, 95))
            arr_normalized = np.nan_to_num(arr_normalized, nan=0.0, posinf=vmax, neginf=0.0)
            arr_normalized = np.clip(arr_normalized, 0, vmax) / vmax
        else:
            arr_normalized = np.zeros_like(arr_normalized)

        rgb_array = np.zeros((self.gridHeight, self.gridWidth, 3), dtype=np.uint8)
        rgb_array[is_river] = [30, 60, 120]
        rgb_array[fire_state == FireState.Burning] = [255, 100, 0]
        rgb_array[fire_state == FireState.Burnt] = [60, 60, 60]

        mask = ~(is_river | (fire_state == FireState.Burning) | (fire_state == FireState.Burnt))
        low_mask = mask & (arr_normalized > 0) & (arr_normalized < 0.33)
        mid_mask = mask & (arr_normalized >= 0.33) & (arr_normalized < 0.66)
        high_mask = mask & (arr_normalized >= 0.66)

        rgb_array[mask & (arr_normalized <= 0)] = [20, 20, 20]

        rgb_array[low_mask, 0] = (arr_normalized[low_mask] * 3 * 180).astype(np.uint8)
        rgb_array[low_mask, 2] = (arr_normalized[low_mask] * 3 * 50).astype(np.uint8)

        mid_t = np.zeros_like(arr_normalized)
        mid_t[mid_mask] = (arr_normalized[mid_mask] - 0.33) / 0.33
        rgb_array[mid_mask, 0] = (180 + mid_t[mid_mask] * 75).astype(np.uint8)
        rgb_array[mid_mask, 1] = (mid_t[mid_mask] * 50).astype(np.uint8)
        rgb_array[mid_mask, 2] = 50

        high_t = np.zeros_like(arr_normalized)
        high_t[high_mask] = (arr_normalized[high_mask] - 0.66) / 0.34
        rgb_array[high_mask, 0] = 255
        rgb_array[high_mask, 1] = (50 + high_t[high_mask] * 100).astype(np.uint8)
        rgb_array[high_mask, 2] = (50 + high_t[high_mask] * 150).astype(np.uint8)
        return rgb_array

    def _render_human(self):
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")

        if self._renderer is None:
            pygame.init()
            self.window_width = 960
            self.window_height = 640
            self._renderer = {}
            self._renderer["screen"] = pygame.display.set_mode((self.window_width, self.window_height))
            self._renderer["clock"] = pygame.time.Clock()
            self._renderer["font"] = pygame.font.Font(None, 28)
            pygame.display.set_caption("Wildfire Environment - FirecastRL")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        rgb_array_scaled = np.repeat(np.repeat(self._build_render_base(), 4, axis=0), 4, axis=1)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_array_scaled, (1, 0, 2)))
        self._renderer["screen"].blit(surface, (0, 0))

        for hx_attack, hy_attack, attack_step in self.helitack_history:
            age = self.step_count - attack_step
            alpha = max(0.2, 1.0 - age / 20.0)
            blue_intensity = int(alpha * 100)
            pygame.draw.circle(self._renderer["screen"], (0, 0, blue_intensity), (hx_attack * 4 + 2, hy_attack * 4 + 2), 8)

        hx, hy = (int(value) for value in self.state["helicopter_coord"])
        heli_screen_x = hx * 4 + 2
        heli_screen_y = hy * 4 + 2
        heli_color = (255, 255, 0)
        heli_outline = (0, 0, 0)

        pygame.draw.line(self._renderer["screen"], heli_outline, (heli_screen_x - 9, heli_screen_y - 9), (heli_screen_x + 9, heli_screen_y + 9), 4)
        pygame.draw.line(self._renderer["screen"], heli_outline, (heli_screen_x + 9, heli_screen_y - 9), (heli_screen_x - 9, heli_screen_y + 9), 4)
        pygame.draw.line(self._renderer["screen"], heli_color, (heli_screen_x - 8, heli_screen_y - 8), (heli_screen_x + 8, heli_screen_y + 8), 2)
        pygame.draw.line(self._renderer["screen"], heli_color, (heli_screen_x + 8, heli_screen_y - 8), (heli_screen_x - 8, heli_screen_y + 8), 2)
        pygame.draw.circle(self._renderer["screen"], heli_outline, (heli_screen_x, heli_screen_y), 5)
        pygame.draw.circle(self._renderer["screen"], heli_color, (heli_screen_x, heli_screen_y), 3)
        pygame.draw.line(self._renderer["screen"], heli_outline, (heli_screen_x - 12, heli_screen_y), (heli_screen_x + 12, heli_screen_y), 3)
        pygame.draw.line(self._renderer["screen"], (200, 200, 200), (heli_screen_x - 11, heli_screen_y), (heli_screen_x + 11, heli_screen_y), 1)

        cells_burning, cells_burnt = self._get_current_fire_stats()
        text = self._renderer["font"].render(
            f"Step: {self.step_count} | Burning: {cells_burning} | Burnt: {cells_burnt}",
            True,
            (255, 255, 255),
            (0, 0, 0),
        )
        self._renderer["screen"].blit(text, (10, 10))
        pygame.display.flip()
        self._renderer["clock"].tick(self.metadata["render_fps"])
        return None

    def _render_rgb_array(self):
        rgb_array_scaled = np.repeat(np.repeat(self._build_render_base(), 4, axis=0), 4, axis=1)

        for hx_attack, hy_attack, attack_step in self.helitack_history:
            age = self.step_count - attack_step
            blue_intensity = int(100 * max(0.2, 1.0 - age / 20.0))
            attack_center_x = hx_attack * 4 + 2
            attack_center_y = hy_attack * 4 + 2
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx * dx + dy * dy <= 64:
                        px = attack_center_x + dx
                        py = attack_center_y + dy
                        if 0 <= px < rgb_array_scaled.shape[1] and 0 <= py < rgb_array_scaled.shape[0]:
                            rgb_array_scaled[py, px] = [0, 0, blue_intensity]

        hx, hy = (int(value) for value in self.state["helicopter_coord"])
        heli_center_x = hx * 4 + 2
        heli_center_y = hy * 4 + 2
        yellow = [255, 255, 0]
        black = [0, 0, 0]

        for offset in range(-8, 9):
            x1, y1 = heli_center_x + offset, heli_center_y + offset
            x2, y2 = heli_center_x + offset, heli_center_y - offset
            for thick in [-1, 0, 1]:
                if 0 <= y1 < rgb_array_scaled.shape[0] and 0 <= x1 < rgb_array_scaled.shape[1] and 0 <= x1 + thick < rgb_array_scaled.shape[1]:
                    rgb_array_scaled[y1, x1 + thick] = black if abs(offset) > 6 else yellow
                if 0 <= y2 < rgb_array_scaled.shape[0] and 0 <= x2 < rgb_array_scaled.shape[1] and 0 <= x2 + thick < rgb_array_scaled.shape[1]:
                    rgb_array_scaled[y2, x2 + thick] = black if abs(offset) > 6 else yellow

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    px = heli_center_x + dx
                    py = heli_center_y + dy
                    if 0 <= px < rgb_array_scaled.shape[1] and 0 <= py < rgb_array_scaled.shape[0]:
                        rgb_array_scaled[py, px] = yellow

        for rx in range(-11, 12):
            px = heli_center_x + rx
            py = heli_center_y
            if 0 <= px < rgb_array_scaled.shape[1] and 0 <= py < rgb_array_scaled.shape[0]:
                rgb_array_scaled[py, px] = [200, 200, 200]

        return rgb_array_scaled

    def close(self):
        if self._renderer is not None:
            try:
                import pygame
                if pygame.get_init():
                    pygame.quit()
            except Exception:
                pass
            self._renderer = None

        self.simulation_running = False
        self.engine = None
        self.state = None