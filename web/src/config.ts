import { ZoneOptions } from "./models/zone";
import { DroughtLevel, Vegetation, TerrainType, VegetationType } from "./types";
import { GENERATED_ASSETS_BASE_URL } from "./env";

interface TownOptions {
  name: string;
  x: number; // [0, 1], position relative to model width
  y: number; // [0, 1], position relative to model height
  terrainType?: TerrainType; // limit town marker to given terrain type
}

export interface ISimulationConfig {
  modelWidth: number; // ft
  modelHeight: number; // ft
  // Note that modelHeight % gridWidth should always be 0!
  gridWidth: number; // ft
  // It will be calculated automatically using model dimensions and grid width.
  readonly gridHeight: number; // ft
  // It will be calculated automatically using model dimensions and grid width.
  readonly cellSize: number; // ft
  // If `elevation` height map is provided, it will be loaded during model initialization and terrain setup dialog
  // won't let users change terrain type. Otherwise, height map URL will be derived from zones `terrainType` properties.
  elevation?: number[][] | string;
  // `unburntIslands` data can provided using image url or 2D array.
  // Otherwise, unburnt islands map URL will be derived from zones `terrainType` properties.
  unburntIslands?: number[][] | string;
  // `zoneIndex` data can provided using image url or 2D array. If it's an array, it should include two or three
  // numbers, depending if model is using two or three zones (0 and 1, or 0, 1, and 2).
  zoneIndex?: number[][] | string;
  // Spark positions, in ft.
  sparks: number[][];
  maxTimeStep: number; // minutes
  // One day in model should last X seconds in real world.
  modelDayInSeconds: number;
  windSpeed: number; // mph
  windDirection: number; // degrees, 0 is northern wind
  neighborsDist: number;
  // In min - note that larger cells will burn the same amount of time. Cell doesn't burn from edge to edge, but
  // its whole area is supposed to burn at the same time. We might consider whether it should be different for
  // different fuel types.
  minCellBurnTime: number;
  // Max elevation of 100% white points in heightmap (image used for elevation data).
  heightmapMaxElevation: number; // ft
  // Number of zones that the model is using. Zones are used to keep properties of some area of the model.
  zonesCount?: 17 | 18;
  zones: ZoneOptions[];
  towns: TownOptions[];
  // Visually fills edges of the terrain by setting elevation to 0.
  fillTerrainEdges: boolean;
  riverData: string | null;
  windScaleFactor: number;
  showModelDimensions: boolean;
  // Time that needs to pass before next fire line can be added.
  fireLineDelay: number;
  // Helitack has a cooldown before it can be used again
  helitackDelay: number;
  maxFireLineLength: number; // ft
  helitackDropRadius: number; // ft
  // Renders burn index.
  showBurnIndex: boolean;
  // Displays alert with current coordinates on mouse click. Useful for authoring.
  showCoordsOnClick: boolean;
  // Number between 0 and 1 which decides how likely is for unburnt island to form (as it's random).
  // 1 means that all the unburnt islands will be visible, 0 means that none of them will be visible.
  unburntIslandProbability: number;
  // Number between 0 and 1 which decides how likely is for a cells to survive fire. Note that there are other factors
  // too. The only vegetation that can survive fire low and medium intensity fire is `Forest`.
  fireSurvivalProbability: number;
  // Locks drought index slider in Terrain Setup dialog.
  droughtIndexLocked: boolean;
  // Makes severe drought option available in Terrain Setup dialog.
  severeDroughtAvailable: boolean;
  // River color, RGBA values (range: [0, 1]). Suggested colors:
  // [0.663,0.855,1,1], [0.337,0.69,0.957,1] or [0.067,0.529,0.882,1]
  riverColor: [number, number, number, number];
  // Authors may want to disable the fireline and helitack features completely
  fireLineAvailable: boolean;
  helitackAvailable: boolean;
  forestWithSuppressionAvailable: boolean;
  // If set to a number, the wind direction and strength will change during the model run.
  changeWindOnDay: number | undefined;
  // Works together with `changeWindOnDay`. Sets the new wind direction (0 to 360). If undefined, it'll be random.
  newWindDirection: number | undefined;
  // Works together with `changeWindOnDay`. Sets the new wind speed (mph). If undefined, it'll be random.
  newWindSpeed: number | undefined;
}

export interface IUrlConfig extends ISimulationConfig {
  preset: string;
}

export const getDefaultConfig: () => IUrlConfig = () => ({
  preset: "default",
  // Most of the presets will use heightmap images that work the best with 120000x80000ft dimensions.
  modelWidth: 120000,
  modelHeight: 80000,
  // 240 works well with presets based on heightmap images.
  // gridWidth: 1200, //HD
  gridWidth: 240, 
  get cellSize() { return this.modelWidth / this.gridWidth; },
  get gridHeight() { return Math.ceil(this.modelHeight / this.cellSize); },
  elevation: undefined, // will be derived from zone properties
  unburntIslands: undefined, // will be derived from zone properties
  zoneIndex: undefined,
  sparks: [],
  maxTimeStep: 180, // minutes
  modelDayInSeconds: 8, // one day in model should last X seconds in real world
  windSpeed: 0, // mph
  windDirection: 0, // degrees, northern wind
  // Note that 0.5 helps to create a nicer, more round shape of neighbours set for a given cell
  // on the rectangular grid when small radius values are used (like 2.5).
  // 2.5 seems to be first value that ensures that fire front looks pretty round.
  // Higher values will make this shape better, but performance will be affected.
  neighborsDist: 2.5,
  minCellBurnTime: 200, // minutes
  // This value works well with existing heightmap images.
  heightmapMaxElevation: 10000,
  // undefined zones count will make them configurable in Terrain Setup dialog.
  zonesCount: 18,
  zones: [
    {
    terrainType: TerrainType.Mountains,
    vegetation: VegetationType.EvergreenNeedleleaf,
    droughtLevel: DroughtLevel.MildDrought // 0. Dummy Zone
  },
  {
    terrainType: TerrainType.Mountains,
    vegetation: VegetationType.EvergreenNeedleleaf,
    droughtLevel: DroughtLevel.MildDrought // 1. Evergreen Needleleaf Forest
  },
  {
    terrainType: TerrainType.Tropical,
    vegetation: VegetationType.EvergreenBroadleaf,
    droughtLevel: DroughtLevel.NoDrought // 2. Evergreen Broadleaf Forest
  },
  {
    terrainType: TerrainType.Plains,
    vegetation: VegetationType.DeciduousNeedleleaf,
    droughtLevel: DroughtLevel.MildDrought // 3. Deciduous Needleleaf Forest
  },
  {
    terrainType: TerrainType.Plains,
    vegetation: VegetationType.DeciduousBroadleaf,
    droughtLevel: DroughtLevel.MildDrought // 4. Deciduous Broadleaf Forest
  },
  {
    terrainType: TerrainType.Hills,
    vegetation: VegetationType.MixedForest,
    droughtLevel: DroughtLevel.MediumDrought // 5. Mixed Forest
  },
  {
    terrainType: TerrainType.Desert,
    vegetation: VegetationType.ClosedShrublands,
    droughtLevel: DroughtLevel.MediumDrought // 6. Closed Shrublands
  },
  {
    terrainType: TerrainType.Desert,
    vegetation: VegetationType.OpenShrublands,
    droughtLevel: DroughtLevel.SevereDrought // 7. Open Shrublands
  },
  {
    terrainType: TerrainType.Plains,
    vegetation: VegetationType.WoodySavannas,
    droughtLevel: DroughtLevel.MildDrought // 8. Woody Savannas
  },
  {
    terrainType: TerrainType.Plains,
    vegetation: VegetationType.Savannas,
    droughtLevel: DroughtLevel.MediumDrought // 9. Savannas
  },
  {
    terrainType: TerrainType.Plains,
    vegetation: VegetationType.Grasslands,
    droughtLevel: DroughtLevel.MildDrought // 10. Grasslands
  },
  {
    terrainType: TerrainType.Wetlands,
    vegetation: VegetationType.PermanentWetlands,
    droughtLevel: DroughtLevel.NoDrought // 11. Permanent Wetlands
  },
  {
    terrainType: TerrainType.Agricultural,
    vegetation: VegetationType.Croplands,
    droughtLevel: DroughtLevel.MildDrought // 12. Croplands
  },
  {
    terrainType: TerrainType.Urban,
    vegetation: VegetationType.UrbanBuilt,
    droughtLevel: DroughtLevel.SevereDrought // 13. Urban and Built-Up
  },
  {
    terrainType: TerrainType.Mixed,
    vegetation: VegetationType.CroplandMosaic,
    droughtLevel: DroughtLevel.MediumDrought // 14. Cropland/Natural Vegetation Mosaic
  },
  {
    terrainType: TerrainType.Ice,
    vegetation: VegetationType.SnowIce,
    droughtLevel: DroughtLevel.NoDrought // 15. Snow and Ice
  },
  {
    terrainType: TerrainType.Desert,
    vegetation: VegetationType.Barren,
    droughtLevel: DroughtLevel.SevereDrought // 16. Barren or Sparsely Vegetated
  },
  {
    terrainType: TerrainType.Water,
    vegetation: VegetationType.Water,
    droughtLevel: DroughtLevel.NoDrought // 17. Water
  }
],
  towns: [],
  fillTerrainEdges: true,
  riverData: `${GENERATED_ASSETS_BASE_URL}/river-texmap.png`,
  windScaleFactor: 0.2, // Note that model is very sensitive to wind.
  // Scale wind values down for now, so changes are less dramatic.
  showModelDimensions: false,
  fireLineDelay: 1440, // a day
  helitackDelay: 240, // four hours
  maxFireLineLength: 15000, // ft
  helitackDropRadius: 2640, // ft (5280 ft = 1 mile)
  showBurnIndex: true,
  showCoordsOnClick: false,
  unburntIslandProbability: 0.5, // [0, 1]
  fireSurvivalProbability: 0.1, // [0, 1]
  droughtIndexLocked: false,
  severeDroughtAvailable: true,
  riverColor: [0.067, 0.529, 0.882, 1],
  fireLineAvailable: true,
  helitackAvailable: true,
  forestWithSuppressionAvailable: true,
  changeWindOnDay: undefined,
  newWindDirection: undefined,
  newWindSpeed: undefined
});

const getURLParam = (name: string) => {
  const url = (self || window).location.href;
  console.log(url);
  
  name = name.replace(/[[]]/g, "\\$&");
  const regex = new RegExp(`[?&]${name}(=([^&#]*)|&|#|$)`);
  const results = regex.exec(url);
  if (!results) return null;
  if (!results[2]) return true;
  return decodeURIComponent(results[2].replace(/\+/g, " "));
};

const isArray = (value: any) => {
  return typeof value === "string" && value.match(/^\[.*\]$/);
};

const isJSON = (value: any) => {
  if (typeof value !== "string") {
    return false;
  }
  try {
    JSON.parse(value);
    return true;
  } catch (e) {
    return false;
  }
};

const getQueryParams = (url: string): { [key: string]: string } => {
  console.log(url);
  
  const params: { [key: string]: any } = {};
  const queryString = url.split('?')[1];
  console.log(queryString);
  
  if (queryString) {
    const pairs = queryString.split('&');
    pairs.forEach(pair => {
      const [key, value] = pair.split('=');
      params[key] = decodeURIComponent(value);
    });
  }
  console.log(params);
  
  return params;
}

export const getUrlConfig: () => IUrlConfig = () => {
  const urlConfig: any = {};
  // Populate `urlConfig` with values read from URL.
  Object.keys(getDefaultConfig()).forEach((key) => {
    const urlValue: any = getURLParam(key);
    
    // const params = getQueryParams((self || window).location.href);
    // const urlValue = params[key];
    // console.log(urlValue);
    
    if (urlValue === true || urlValue === "true") {
      urlConfig[key] = true;
    } else if (urlValue === "false") {
      urlConfig[key] = false;
    } else if (isJSON(urlValue)) {
      urlConfig[key] = JSON.parse(urlValue);
    } else if (isArray(urlValue)) {
      // Array can be provided in URL using following format:
      // &parameter=[value1,value2,value3]
      if (urlValue === "[]") {
        urlConfig[key] = [];
      } else {
        urlConfig[key] = urlValue.substring(1, urlValue.length - 1).split(",");
      }
    } else if (urlValue !== null && !isNaN(urlValue)) {
      // !isNaN(string) means isNumber(string).
      urlConfig[key] = parseFloat(urlValue);
    } else if (urlValue !== null) {
      urlConfig[key] = urlValue;
    }
  });
  return urlConfig as IUrlConfig;
};

