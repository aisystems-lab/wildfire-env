import math
import random

from ..environment.array_state import (
    ArrayFireState,
    burn_index_for_values,
    can_survive_fire,
    is_burnable_for_index,
)
from ..environment.enums import BurnIndex, FireState
from ..environment.wind import Wind
from .fire_spread_rate import get_fire_spread_rate_from_arrays
from .utils import for_each_point_between

model_day = 1440  # minutes

end_of_low_intensity_fire_probability = {0: 0.0, 1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.5, 6: 0.7, 7: 1.0}


class FireEngine:
    def __init__(self, wind: Wind, config):
        # self.cells : List[Cell] = cells
        self.wind: Wind = wind
        self.grid_width = config.gridWidth
        self.grid_height = config.gridHeight
        self.cell_size = config.cellSize
        self.min_cell_burn_time = config.minCellBurnTime
        self.neighbors_dist = config.neighborsDist
        self.fire_survival_probability = config.fireSurvivalProbability
        self.end_of_low_intensity_fire = False
        self.fire_did_stop = False
        self.day = 0
        self.burned_cells_in_zone = {}
        self._spread_offsets = self._build_spread_offsets()

    def _build_spread_offsets(self):
        offsets = []
        max_offset = int(math.ceil(self.neighbors_dist))
        for dy in range(-max_offset, max_offset + 1):
            for dx in range(-max_offset, max_offset + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy > self.neighbors_dist**2:
                    continue

                line_points = []

                def callback(x, y, _, lp=line_points):
                    lp.append((x, y))

                for_each_point_between(0, 0, dx, dy, callback)
                offsets.append((dx, dy, tuple(line_points[1:-1])))
        return offsets

    def _iter_burnable_neighbors(
        self,
        state: ArrayFireState,
        x0: int,
        y0: int,
        source_burn_index: int,
    ):
        nonburnable = state.is_nonburnable
        fire_lines = state.is_fire_line

        for dx, dy, between_points in self._spread_offsets:
            nx = x0 + dx
            ny = y0 + dy
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                continue

            if not is_burnable_for_index(
                bool(nonburnable[ny, nx]),
                bool(fire_lines[ny, nx]),
                source_burn_index,
            ):
                continue

            blocked = False
            for px, py in between_points:
                ix = x0 + px
                iy = y0 + py
                if not (0 <= ix < self.grid_width and 0 <= iy < self.grid_height):
                    blocked = True
                    break
                if not is_burnable_for_index(
                    bool(nonburnable[iy, ix]),
                    bool(fire_lines[iy, ix]),
                    source_burn_index,
                ):
                    blocked = True
                    break

            if not blocked:
                yield nx, ny

    def update_fire_array(self, state: ArrayFireState, time: float) -> None:
        new_day = int(time // model_day)
        if new_day != self.day:
            self.day = new_day
            if random.random() <= end_of_low_intensity_fire_probability.get(new_day, 0.0):
                self.end_of_low_intensity_fire = True

        new_ignition_data: dict[tuple[int, int], float] = {}
        new_fire_state_data: dict[tuple[int, int], int] = {}
        self.fire_did_stop = True

        ignition = state.ignition_time
        fire_state = state.fire_state
        burn_time = state.burn_time
        spread_rate = state.spread_rate
        survivors = state.is_fire_survivor
        zone_idx = state.zone_idx
        vegetation = state.vegetation
        elevation = state.elevation
        moisture_content = state.moisture_content

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                current_state = int(fire_state[y, x])
                ignition_time = float(ignition[y, x])

                if current_state == FireState.Burning or (
                    current_state == FireState.Unburnt and ignition_time < math.inf
                ):
                    self.fire_did_stop = False

                if current_state == FireState.Burning and time - ignition_time > float(burn_time[y, x]):
                    new_fire_state_data[(y, x)] = FireState.Burnt
                    source_burn_index = burn_index_for_values(int(vegetation[y, x]), float(spread_rate[y, x]))
                    if (
                        can_survive_fire(int(vegetation[y, x]), source_burn_index)
                        and random.random() < self.fire_survival_probability
                    ):
                        survivors[y, x] = True

                elif current_state == FireState.Unburnt and time > ignition_time:
                    new_fire_state_data[(y, x)] = FireState.Burning
                    zone = int(zone_idx[y, x])
                    self.burned_cells_in_zone[zone] = self.burned_cells_in_zone.get(zone, 0) + 1

                    source_burn_index = burn_index_for_values(int(vegetation[y, x]), float(spread_rate[y, x]))
                    fire_should_spread = not self.end_of_low_intensity_fire or source_burn_index != BurnIndex.Low
                    if not fire_should_spread:
                        continue

                    for nx, ny in self._iter_burnable_neighbors(state, x, y, source_burn_index):
                        target_state = int(fire_state[ny, nx])
                        target_spread_rate = get_fire_spread_rate_from_arrays(
                            source_x=x,
                            source_y=y,
                            target_x=nx,
                            target_y=ny,
                            source_elevation=float(elevation[y, x]),
                            target_elevation=float(elevation[ny, nx]),
                            target_vegetation=int(vegetation[ny, nx]),
                            target_moisture_content=float(moisture_content[ny, nx]),
                            wind=self.wind,
                            cell_size=self.cell_size,
                        )
                        ignition_delta = (
                            math.dist((x, y), (nx, ny)) * self.cell_size / target_spread_rate
                            if target_spread_rate != 0
                            else float("inf")
                        )

                        if target_state == FireState.Unburnt:
                            current_ignition = new_ignition_data.get((ny, nx), float(ignition[ny, nx]))
                            new_ignition = min(ignition_time + ignition_delta, current_ignition)
                            new_ignition_data[(ny, nx)] = new_ignition

                            new_burn_time = (new_ignition - ignition_time) + self.min_cell_burn_time
                            if new_burn_time < float(burn_time[ny, nx]):
                                burn_time[ny, nx] = new_burn_time
                            if target_spread_rate > float(spread_rate[ny, nx]):
                                spread_rate[ny, nx] = target_spread_rate

        for (y, x), state_value in new_fire_state_data.items():
            fire_state[y, x] = state_value
        for (y, x), time_value in new_ignition_data.items():
            ignition[y, x] = time_value
