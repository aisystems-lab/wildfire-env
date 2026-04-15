from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .cell import FIRE_LINE_DEPTH, MAX_BURN_TIME, moisture_lookup_by_land_cover
from .enums import BurnIndex, DroughtLevel, FireState, VegetationType

NON_BURNABLE_ZONE_INDICES = {12, 14, 16, 17, 18}


def _build_moisture_lookup() -> np.ndarray:
    table = np.full((18, 4), np.inf, dtype=np.float32)
    for vegetation, values in moisture_lookup_by_land_cover.items():
        table[int(vegetation.value), :] = np.asarray(values, dtype=np.float32)
    return table


MOISTURE_LOOKUP = _build_moisture_lookup()


def burn_index_for_values(vegetation: int, spread_rate: float) -> int:
    if vegetation == VegetationType.Grasslands.value:
        return BurnIndex.Low if spread_rate < 45 else BurnIndex.Medium

    if vegetation in (
        VegetationType.ClosedShrublands.value,
        VegetationType.OpenShrublands.value,
    ):
        if spread_rate < 10:
            return BurnIndex.Low
        if spread_rate < 50:
            return BurnIndex.Medium
        return BurnIndex.High

    if vegetation in (
        VegetationType.EvergreenNeedleleaf.value,
        VegetationType.DeciduousNeedleleaf.value,
        VegetationType.MixedForest.value,
        VegetationType.EvergreenBroadleaf.value,
        VegetationType.DeciduousBroadleaf.value,
    ):
        return BurnIndex.Low if spread_rate < 25 else BurnIndex.Medium

    return BurnIndex.Low


def can_survive_fire(vegetation: int, burn_index: int) -> bool:
    return burn_index == BurnIndex.Low and vegetation == VegetationType.PermanentWetlands.value


def is_burnable_for_index(
    is_nonburnable: bool,
    is_fire_line: bool,
    burn_index: int,
) -> bool:
    return (not is_nonburnable) and (not is_fire_line or burn_index == BurnIndex.High)


@dataclass
class ArrayFireState:
    width: int
    height: int
    zone_idx: np.ndarray
    vegetation: np.ndarray
    base_drought: np.ndarray
    base_elevation: np.ndarray
    is_river: np.ndarray
    is_unburnt_island: np.ndarray
    is_fire_line: np.ndarray
    is_fire_line_under_construction: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    ignition_time: np.ndarray
    spread_rate: np.ndarray
    burn_time: np.ndarray
    fire_state: np.ndarray
    helitack_drops: np.ndarray
    is_fire_survivor: np.ndarray

    @classmethod
    def from_grid_inputs(
        cls,
        width: int,
        height: int,
        zones: Sequence[object],
        zone_index_flat,
        elevation_flat,
        *,
        fill_terrain_edges: bool = False,
    ) -> "ArrayFireState":
        zone_idx = np.asarray(zone_index_flat, dtype=np.int16).reshape(height, width)
        zone_idx = np.clip(zone_idx, 0, max(0, len(zones) - 1))

        if elevation_flat is None:
            base_elevation = np.zeros((height, width), dtype=np.float32)
        else:
            base_elevation = np.asarray(elevation_flat, dtype=np.float32).reshape(height, width)

        if fill_terrain_edges:
            base_elevation = base_elevation.copy()
            base_elevation[0, :] = 0.0
            base_elevation[-1, :] = 0.0
            base_elevation[:, 0] = 0.0
            base_elevation[:, -1] = 0.0

        zone_vegetation_lookup = np.asarray(
            [int(getattr(zone.vegetation, "value", 0)) for zone in zones],
            dtype=np.int16,
        )
        zone_drought_lookup = np.asarray(
            [int(getattr(zone.droughtLevel, "value", 0)) for zone in zones],
            dtype=np.int8,
        )

        vegetation = zone_vegetation_lookup[zone_idx]
        base_drought = zone_drought_lookup[zone_idx]
        is_river = np.isin(zone_idx, tuple(NON_BURNABLE_ZONE_INDICES))
        is_unburnt_island = np.zeros((height, width), dtype=bool)
        is_fire_line = np.zeros((height, width), dtype=bool)
        is_fire_line_under_construction = np.zeros((height, width), dtype=bool)

        x_coords = np.broadcast_to(np.arange(width, dtype=np.int16), (height, width))
        y_coords = np.broadcast_to(np.arange(height, dtype=np.int16)[:, None], (height, width))

        state = cls(
            width=width,
            height=height,
            zone_idx=zone_idx,
            vegetation=vegetation,
            base_drought=base_drought,
            base_elevation=base_elevation,
            is_river=is_river,
            is_unburnt_island=is_unburnt_island,
            is_fire_line=is_fire_line,
            is_fire_line_under_construction=is_fire_line_under_construction,
            x_coords=x_coords,
            y_coords=y_coords,
            ignition_time=np.empty((height, width), dtype=np.float32),
            spread_rate=np.empty((height, width), dtype=np.float32),
            burn_time=np.empty((height, width), dtype=np.float32),
            fire_state=np.empty((height, width), dtype=np.int8),
            helitack_drops=np.empty((height, width), dtype=np.int16),
            is_fire_survivor=np.empty((height, width), dtype=bool),
        )
        state.reset_dynamic()
        return state

    def reset_dynamic(self) -> None:
        self.ignition_time.fill(np.inf)
        self.spread_rate.fill(0.0)
        self.burn_time.fill(float(MAX_BURN_TIME))
        self.fire_state.fill(FireState.Unburnt)
        self.helitack_drops.fill(0)
        self.is_fire_survivor.fill(False)
        self.is_fire_line.fill(False)
        self.is_fire_line_under_construction.fill(False)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def is_nonburnable(self) -> np.ndarray:
        return np.logical_or(self.is_river, self.is_unburnt_island)

    @property
    def drought_level(self) -> np.ndarray:
        drops = np.minimum(self.helitack_drops, DroughtLevel.SevereDrought.value).astype(np.int8, copy=False)
        return np.maximum(self.base_drought - drops, DroughtLevel.NoDrought.value).astype(np.int8, copy=False)

    @property
    def elevation(self) -> np.ndarray:
        return self.base_elevation - self.is_fire_line.astype(np.float32) * FIRE_LINE_DEPTH

    @property
    def moisture_content(self) -> np.ndarray:
        moisture = MOISTURE_LOOKUP[self.vegetation, self.drought_level]
        return np.where(self.is_nonburnable, np.inf, moisture).astype(np.float32, copy=False)
