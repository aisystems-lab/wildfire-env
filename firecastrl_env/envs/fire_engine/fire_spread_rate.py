import math
from dataclasses import dataclass
import numpy as np
from ..environment.enums import VegetationType
from ..environment.vector import Vector2

@dataclass
class WindProps:
    speed:float
    direction:float

@dataclass
class Fuel:
    sav: float
    net_fuel_load: float
    fuel_bed_depth: float
    packing_ratio: float
    mx: float

FuelConstants = {
    # True surface fuels
    VegetationType.Grasslands: Fuel(2100, 0.294, 3.0, 0.00306, 0.15),
    VegetationType.ClosedShrublands: Fuel(1672, 0.239, 1.2, 0.01198, 0.30),
    VegetationType.OpenShrublands: Fuel(1500, 0.20, 1.0, 0.010, 0.30),

    # Forest litter
    VegetationType.EvergreenNeedleleaf: Fuel(1716, 0.0459, 0.10, 0.04878, 0.20),
    VegetationType.DeciduousNeedleleaf: Fuel(1650, 0.040, 0.10, 0.045, 0.20),
    VegetationType.EvergreenBroadleaf: Fuel(1500, 0.06, 0.12, 0.035, 0.25),
    VegetationType.DeciduousBroadleaf: Fuel(1400, 0.05, 0.10, 0.030, 0.25),
    VegetationType.MixedForest: Fuel((1716 + 1500) / 2, (0.0459 + 0.06) / 2, 0.11, 0.042, 0.22),

    # Savanna / woody grassland
    VegetationType.WoodySavannas: Fuel(1400, 0.18, 0.8, 0.008, 0.30),
    VegetationType.Savannas: Fuel(1450, 0.17, 0.7, 0.0085, 0.25),

    # Agricultural
    VegetationType.Croplands: Fuel(1200, 0.20, 0.50, 0.015, 0.30),
    VegetationType.CroplandMosaic: Fuel(1300, 0.18, 0.60, 0.012, 0.25),

    # Wet or sparsely vegetated
    VegetationType.PermanentWetlands: Fuel(800, 0.12, 0.50, 0.007, 0.50),
    VegetationType.Barren: Fuel(600, 0.01, 0.05, 0.002, 0.10),

    # Non-burnable
    VegetationType.UrbanBuilt: Fuel(0, 0, 0, 0, 0),
    VegetationType.SnowIce: Fuel(0, 0, 0, 0, 0),
    VegetationType.Water: Fuel(0, 0, 0, 0, 0),
}

FUEL_LOOKUP = np.zeros((18, 5), dtype=np.float32)
for vegetation, fuel in FuelConstants.items():
    FUEL_LOOKUP[int(vegetation.value)] = np.asarray(
        [
            fuel.sav,
            fuel.net_fuel_load,
            fuel.fuel_bed_depth,
            fuel.packing_ratio,
            fuel.mx,
        ],
        dtype=np.float32,
    )

UNBURNABLE_VEGETATION = {
    VegetationType.UrbanBuilt,
    VegetationType.SnowIce,
    VegetationType.Water
}
UNBURNABLE_VEGETATION_VALUES = {vegetation.value for vegetation in UNBURNABLE_VEGETATION}

heat_content = 8000
total_mineral_content = 0.0555
effective_mineral_content = 0.01

def get_direction_factor_from_coords(
    source_x: int,
    source_y: int,
    target_x: int,
    target_y: int,
    effective_wind_speed: float,
    max_spread_direction: float,
) -> float:
    effective_wind_speed_mph = effective_wind_speed / 88.0
    z_term = 1 + 0.25 * effective_wind_speed_mph
    e_term = math.sqrt(z_term ** 2 - 1) / z_term

    cell_vector = Vector2(target_x - source_x, target_y - source_y)
    relative_angle = abs(cell_vector.angle() - max_spread_direction)
    return (1 - e_term) / (1 - e_term * math.cos(relative_angle))


def get_fire_spread_rate_from_arrays(
    *,
    source_x: int,
    source_y: int,
    target_x: int,
    target_y: int,
    source_elevation: float,
    target_elevation: float,
    target_vegetation: int,
    target_moisture_content: float,
    wind: WindProps,
    cell_size: float,
) -> float:
    if target_vegetation <= 0 or target_vegetation >= len(FUEL_LOOKUP):
        return 0.0

    if target_vegetation in UNBURNABLE_VEGETATION_VALUES:
        return 0.0

    sav, net_fuel_load, fuel_bed_depth, packing_ratio, mx = FUEL_LOOKUP[target_vegetation]
    if sav <= 0 or net_fuel_load <= 0 or fuel_bed_depth <= 0 or packing_ratio <= 0 or mx <= 0:
        return 0.0

    moisture_content_ratio = float(target_moisture_content) / float(mx)
    sav_factor = math.pow(float(sav), 1.5)

    a_term = 133 * math.pow(float(sav), -0.7913)
    b_term = 0.02526 * math.pow(float(sav), 0.54)
    c_term = 7.47 * math.exp(-0.133 * math.pow(float(sav), 0.55))
    e_term = 0.715 * (-0.000359 * float(sav))

    max_reaction_velocity = sav_factor / (495 + (0.0594 * sav_factor))
    optimum_packing_ratio = 3.348 * math.pow(float(sav), -0.8189)
    ratio = float(packing_ratio) / optimum_packing_ratio
    if ratio <= 0:
        return 0.0

    optimum_reaction_velocity = max_reaction_velocity * math.pow(ratio, a_term) * math.exp(a_term * (1 - ratio))

    moisture_damping = (
        1
        - (2.59 * moisture_content_ratio)
        + (5.11 * math.pow(moisture_content_ratio, 2))
        - (3.52 * math.pow(moisture_content_ratio, 3))
    )

    mineral_damping = 0.174 * math.pow(effective_mineral_content, -0.19)
    reaction_intensity = optimum_reaction_velocity * float(net_fuel_load) * heat_content * moisture_damping * mineral_damping

    propagating_flux = 1 / (192 + (0.2595 * float(sav))) * math.exp(
        (0.792 + (0.681 * math.sqrt(float(sav)))) * (float(packing_ratio) + 0.1)
    )
    fuel_load = float(net_fuel_load) / (1 - total_mineral_content)
    bulk_density = fuel_load / float(fuel_bed_depth)
    effective_heating_number = math.exp(-138 / float(sav))
    heat_pre_ignition = 250 + (1116 * float(target_moisture_content))

    r0 = reaction_intensity * propagating_flux / (bulk_density * effective_heating_number * heat_pre_ignition)

    wind_speed_ft_per_min = wind.speed * 88
    wind_factor = c_term * math.pow(wind_speed_ft_per_min, b_term) * math.pow(ratio, -e_term)

    dx = target_x - source_x
    dy = target_y - source_y
    if dx == 0 or dy == 0:
        dist_in_ft = abs(dx + dy) * cell_size
    else:
        dist_in_ft = math.sqrt(dx ** 2 + dy ** 2) * cell_size

    elevation_diff = target_elevation - source_elevation
    slope_tan = elevation_diff / dist_in_ft if dist_in_ft != 0 else 0.0
    slope_factor = 5.275 * math.pow(float(packing_ratio), -0.3) * math.pow(slope_tan, 2)

    origin = Vector2(0, 0)
    wind_vector = Vector2(0, -1).rotateAround(origin, -math.radians(wind.direction))
    if target_elevation >= source_elevation:
        upslope_vector = Vector2(dx, dy)
    else:
        upslope_vector = Vector2(-dx, -dy)

    dw = r0 * wind_factor
    wind_vector.setLength(dw)

    ds = r0 * slope_factor
    upslope_vector.setLength(ds)

    max_spread_vector = wind_vector.add(upslope_vector)
    rh = r0 + max_spread_vector.length()
    effective_wind_factor = rh / r0 - 1

    denominator = c_term * math.pow(ratio, -e_term)
    if denominator == 0:
        return 0.0

    base = effective_wind_factor / denominator
    if base <= 0:
        return 0.0

    effective_wind_speed = math.pow(base, 1 / b_term)
    direction_factor = get_direction_factor_from_coords(
        source_x,
        source_y,
        target_x,
        target_y,
        effective_wind_speed,
        max_spread_vector.angle(),
    )

    return rh * direction_factor

