from io import BytesIO
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from ..environment.enums import FireState
from .. import config
from ..environment.array_state import ArrayFireState

def populate_grid(width: int, height: int, image: List[List[float]], interpolate: bool = False) -> List[float]:
    arr = []

    image_height = len(image)
    image_width = len(image[0])

    num_grid_cells_per_image_row_pixel = (
        (image_height - 1) / (height - 1) if interpolate else image_height / height
    )
    num_grid_cells_per_image_col_pixel = (
        (image_width - 1) / (width - 1) if interpolate else image_width / width
    )

    image_row_index = image_height - 1
    image_row_advance = 0

    for r in range(height):
        image_col_index = 0
        image_col_advance = 0

        for c in range(width):
            value = image[image_row_index][image_col_index]

            if interpolate:
                bottom_left = image[image_row_index][image_col_index]
                bottom_right = (
                    image[image_row_index][image_col_index + 1]
                    if image_col_index + 1 < image_width
                    else bottom_left
                )
                top_left = (
                    image[image_row_index - 1][image_col_index]
                    if image_row_index - 1 >= 0
                    else bottom_left
                )
                top_right = (
                    image[image_row_index - 1][image_col_index + 1]
                    if image_row_index - 1 >= 0 and image_col_index + 1 < image_width
                    else top_left if image_row_index - 1 >= 0 else bottom_right
                )

                value = (
                    bottom_left * (1 - image_col_advance) * (1 - image_row_advance) +
                    bottom_right * image_col_advance * (1 - image_row_advance) +
                    top_left * (1 - image_col_advance) * image_row_advance +
                    top_right * image_col_advance * image_row_advance
                )

            arr.append(value)

            image_col_advance += num_grid_cells_per_image_col_pixel
            if image_col_advance >= 1:
                advance = int(image_col_advance)
                image_col_index += advance
                image_col_advance -= advance

        image_row_advance += num_grid_cells_per_image_row_pixel
        if image_row_advance >= 1:
            advance = int(image_row_advance)
            image_row_index -= advance
            image_row_advance -= advance

    return arr

def get_image_data(
    img_src: str,
    map_color: Callable[[Tuple[int, int, int, int]], int],
) -> List[List[int]]:
    try:
        if img_src.startswith("http"):
            response = requests.get(img_src)
            img = Image.open(BytesIO(response.content))
        else:
            CURRENT_DIR = os.path.dirname(__file__)
            image_path = os.path.join(CURRENT_DIR, '..', 'training_environments', img_src)
            img = Image.open(image_path)

        img = img.convert("RGBA")
        raw_data = np.array(img)
        img.close()

    except Exception as e:
        raise RuntimeError(f"Cannot load image {img_src}: {e}")

    height, width, _ = raw_data.shape
    result: List[List[int]] = [
        [map_color(tuple(raw_data[y, x])) for x in range(width)]
        for y in range(height)
    ]

    return result

def get_input_data(
    input_data: Optional[str],
    grid_width: int,
    grid_height: int,
    interpolate: bool,
    map_color: Callable[[Tuple[int, int, int, int]], int],
) -> Optional[np.ndarray]:
    """
    Converts image or grid to resized/interpolated numpy array using map_color
    """
    if input_data is None:
        return None

    try:
        image_data = get_image_data(input_data, map_color)
        return populate_grid(grid_width, grid_height, image_data, interpolate)
    except Exception as e:
        print(f"Error in get_input_data: {e}")
        return None
    
def get_elevation_data(
    config: Dict[str, Any],
    elevationImage
) -> Optional[np.ndarray]:
    """
    Loads elevation data from an image, where R and G channels encode 16-bit elevation.
    """
    # elevation = config.get("elevation", "data/heightmap_1200x813_2.png")
    heightmap_max = getattr(config,'heightmapMaxElevation', 10000)
    grid_width = getattr(config,'gridWidth', 50)
    grid_height = getattr(config,'gridHeight', 50)

    def height_fn(rgba: Tuple[int, int, int, int]) -> float:
        high_byte = rgba[0]
        low_byte = rgba[1]
        value16 = (high_byte << 8) | low_byte
        h_norm = value16 / 65535
        return h_norm * heightmap_max

    return get_input_data(
        input_data=elevationImage,
        grid_width=grid_width,
        grid_height=grid_height,
        interpolate=True,
        map_color=height_fn
    )

def get_land_cover_zone_index(config: dict, landcoverImage) -> Optional[np.ndarray]:

    # RGB string to land cover index (MODIS IGBP)
    rgb_to_land_cover_index: Dict[str, int] = {
        "0,100,0": 1,       # Evergreen Needleleaf Forest
        "34,139,34": 2,     # Evergreen Broadleaf Forest
        "50,205,50": 3,     # Deciduous Needleleaf Forest
        "0,128,0": 4,       # Deciduous Broadleaf Forest
        "60,179,113": 5,    # Mixed Forest
        "240,230,140": 6,   # Closed Shrublands
        "218,165,32": 7,    # Open Shrublands
        "128,128,0": 8,     # Woody Savannas
        "154,205,50": 9,    # Savannas
        "144,238,144": 10,  # Grasslands
        "143,188,143": 11,  # Permanent Wetlands
        "210,180,140": 12,  # Croplands
        "128,128,128": 13,  # Urban and Built-Up
        "222,184,135": 14,  # Cropland/Natural Vegetation Mosaic
        "255,255,255": 15,  # Snow and Ice
        "211,211,211": 16,  # Barren or Sparsely Vegetated
        "0,0,255": 17       # Water
    }

    def map_rgb(rgba: Tuple[int, int, int, int]) -> int:
        key = f"{rgba[0]},{rgba[1]},{rgba[2]}"
        return rgb_to_land_cover_index.get(key, 1)  # default to 1 if unknown

    return get_input_data(
        input_data=landcoverImage,
        grid_width = getattr(config, "gridWidth", 50),
        grid_height = getattr(config, "gridHeight", 50),
        interpolate=False,
        map_color=map_rgb
    )

def get_grid_index_for_location(grid_x: int, grid_y: int, width: int) -> int:
    return grid_x + grid_y * width

def create_array_fire_state(env_id: int, zones) -> ArrayFireState:
    landcover_image_name = f"landcover_{env_id + 1}.png"
    elevation_image_name = f"heightmap_{env_id + 1}.png"
    zone_index = get_land_cover_zone_index(config, landcover_image_name)
    elevation = get_elevation_data(config, elevation_image_name)

    if zone_index is None:
        zone_index = np.zeros(config.gridWidth * config.gridHeight, dtype=np.int16)

    return ArrayFireState.from_grid_inputs(
        width=config.gridWidth,
        height=config.gridHeight,
        zones=zones,
        zone_index_flat=zone_index,
        elevation_flat=elevation,
        fill_terrain_edges=config.fillTerrainEdges,
    )


def perform_helitack_array(state: ArrayFireState, array_x: int, array_y: int) -> int:
    grid_width = state.width
    grid_height = state.height
    if (
        array_x < 0 or array_x >= grid_width or
        array_y < 0 or array_y >= grid_height
    ):
        return 0

    radius = round(getattr(config, "helitackDropRadius", 50) / getattr(config, "cellSize", 50))
    quenched_cells = 0

    for x in range(array_x - radius, array_x + radius):
        for y in range(array_y - radius, array_y + radius + 1):
            if (x - array_x) ** 2 + (y - array_y) ** 2 <= radius ** 2:
                next_cell_x = array_x - (x - array_x)
                next_cell_y = array_y - (y - array_y)

                if 0 <= next_cell_x < grid_width and 0 <= next_cell_y < grid_height:
                    state.helitack_drops[next_cell_y, next_cell_x] += 1
                    state.ignition_time[next_cell_y, next_cell_x] = np.inf

                    if state.fire_state[next_cell_y, next_cell_x] == FireState.Burning:
                        state.fire_state[next_cell_y, next_cell_x] = FireState.Unburnt
                        quenched_cells += 1

    return quenched_cells

def is_helicopter_on_fire(fire_status_list: np.ndarray, array_x: int, array_y: int) -> bool:
    # Add bounds checking
    if (
        fire_status_list.size == 0 or
        array_y < 0 or array_y >= len(fire_status_list) or
        array_x < 0 or array_x >= len(fire_status_list[0])
    ):
        print(f"Invalid coordinates: x={array_x}, y={array_y}, map size={len(fire_status_list)}x{len(fire_status_list[0]) if fire_status_list else 0}")
        return False

    fire_status = fire_status_list[array_y][array_x]
    normalized_burning = (FireState.Burning + 1) / 9.0
    # fire_state = fire_status // 3  
    # return fire_state == FireState.Burning
    return np.isclose(fire_status, normalized_burning, atol=1e-5)