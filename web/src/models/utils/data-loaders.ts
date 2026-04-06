import { TerrainType } from "../../types";
import { ISimulationConfig } from "../../config";
import { getInputData } from "./image-utils";
import { Zone } from "../zone";
import { log } from "console";
import { GENERATED_ASSETS_BASE_URL } from "../../env";

// Maps zones config to image data files (see data dir)
const zonesToImageDataFile = (zones: Zone[]): string => {
  const zoneTypes = zones.map(z => TerrainType[z.terrainType].toLowerCase());
  return `${GENERATED_ASSETS_BASE_URL}/${zoneTypes.join("-")}`;
};

export const getZoneIndex = (
  config: ISimulationConfig,
  zoneIndex: number[][] | string
): Promise<number[] | undefined> => {
  return getInputData(
    zoneIndex,
    config.gridWidth,
    config.gridHeight,
    false,
    (rgba: [number, number, number, number]) => {
      // Red is zone 1, green is zone 2, blue is zone 3
      if (rgba[0] >= rgba[1] && rgba[0] >= rgba[2]) return 0;
      if (rgba[1] >= rgba[0] && rgba[1] >= rgba[2]) return 1;
      return 2;
    }
  );
};

export const getLandCoverZoneIndex = (
  config: ISimulationConfig
): Promise<number[] | undefined> => {
  const landcoverImage =  `${GENERATED_ASSETS_BASE_URL}/landcover_1200x813.png`;

  // Map MODIS IGBP landcover classes (based on color palette) to raw landcover index (1-17)
  const rgbToLandCoverIndex: Record<string, number> = {
    "0,100,0": 1,       // Evergreen Needleleaf Forest
    "34,139,34": 2,     // Evergreen Broadleaf Forest
    "50,205,50": 3,     // Deciduous Needleleaf Forest
    "0,128,0": 4,       // Deciduous Broadleaf Forest
    "60,179,113": 5,    // Mixed Forest
    "240,230,140": 6,   // Closed Shrublands
    "218,165,32": 7,    // Open Shrublands
    "128,128,0": 8,     // Woody Savannas
    "154,205,50": 9,    // Savannas
    "144,238,144": 10,  // Grasslands
    "143,188,143": 11,  // Permanent Wetlands
    "210,180,140": 12,  // Croplands
    "128,128,128": 13,  // Urban and Built-Up
    "222,184,135": 14,  // Cropland/Natural Vegetation Mosaic
    "255,255,255": 15,  // Snow and Ice
    "211,211,211": 16,  // Barren or Sparsely Vegetated
    "0,0,255": 17       // Water
  };

  return getInputData(
    landcoverImage,
    config.gridWidth,
    config.gridHeight,
    true,
    (rgba: [number, number, number, number]) => {
      const key = `${rgba[0]},${rgba[1]},${rgba[2]}`;
      return rgbToLandCoverIndex[key] ?? 1; // return 0 if unknown color
    }
  );
};

export const getElevationData = (
  config: ISimulationConfig,
  zones: Zone[]
): Promise<number[] | undefined> => {
  let elevation = config.elevation;
  console.log(elevation);

  if (!elevation) {
    elevation = `${GENERATED_ASSETS_BASE_URL}/heightmap_1200x813_2.png`;
  }

  const heightFn = (rgba: [number, number, number, number]) => {
    const highByte = rgba[0];
    const lowByte = rgba[1];
    const value16 = (highByte << 8) | lowByte;
    const hNorm = value16 / 65535;
    return hNorm * config.heightmapMaxElevation;
  };

  return getInputData(
    elevation,
    config.gridWidth,
    config.gridHeight,
    true,
    heightFn
  );
};

export const getUnburntIslandsData = (
  config: ISimulationConfig,
  zones: Zone[]
): Promise<number[] | undefined> => {
  let islandsFile = config.unburntIslands;
  if (!islandsFile) {
    islandsFile = `${zonesToImageDataFile(zones)}-islands.png`;
  }

  const islandActive: Record<number, number> = {};
  return getInputData(
    islandsFile,
    config.gridWidth,
    config.gridHeight,
    true,
    (rgba: [number, number, number, number]) => {
      const shade = rgba[0];
      if (shade < 255) {
        if (islandActive[shade] === undefined) {
          islandActive[shade] = Math.random() < config.unburntIslandProbability ? 1 : 0;
        }
        return islandActive[shade];
      }
      return 0;
    }
  );
};

export const getRiverData = (
  config: ISimulationConfig
): Promise<number[] | undefined> => {
  if (!config.riverData) return Promise.resolve(undefined);

  return getInputData(
    config.riverData,
    config.gridWidth,
    config.gridHeight,
    true,
    (rgba: [number, number, number, number]) => (rgba[3] > 0 ? 1 : 0)
  );
};
