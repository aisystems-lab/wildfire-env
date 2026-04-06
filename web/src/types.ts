import { Vector2 } from "three";

export interface Fuel {
  sav: number;
  packingRatio: number;
  netFuelLoad: number;
  mx: number;
  fuelBedDepth: number;
}

export interface Town {
  name: string;
  position: Vector2;
}

export enum Vegetation {
  Grass = 0,
  Shrub = 1,
  Forest = 2,
  ForestWithSuppression = 3
}

export const vegetationLabels: Record<Vegetation, string> = {
  [Vegetation.Grass]: "Grass",
  [Vegetation.Shrub]: "Shrub",
  [Vegetation.Forest]: "Forest",
  [Vegetation.ForestWithSuppression]: "Forest With Suppression"
};

export enum TerrainType {
  Mountains = "mountains",
  Plains = "plains",
  Hills = "hills",
  Tropical = "tropical",
  Desert = "desert",
  Wetlands = "wetlands",
  Agricultural = "agricultural",
  Urban = "urban",
  Mixed = "mixed",
  Ice = "ice",
  Water = "water"
}


export const terrainLabels: Record<TerrainType, string> = {
  [TerrainType.Plains]: "Plains",
  [TerrainType.Foothills]: "Foothills",
  [TerrainType.Mountains]: "Mountains",
};

export enum DroughtLevel {
  NoDrought = 0,
  MildDrought = 1,
  MediumDrought = 2,
  SevereDrought = 3
}

export const droughtLabels: Record<DroughtLevel, string> = {
  [DroughtLevel.NoDrought]: "No Drought",
  [DroughtLevel.MildDrought]: "Mild Drought",
  [DroughtLevel.MediumDrought]: "Medium Drought",
  [DroughtLevel.SevereDrought]: "Severe Drought",
};

export enum VegetationType {
  EvergreenNeedleleaf = 1,
  EvergreenBroadleaf = 2,
  DeciduousNeedleleaf = 3,
  DeciduousBroadleaf = 4,
  MixedForest = 5,
  ClosedShrublands = 6,
  OpenShrublands = 7,
  WoodySavannas = 8,
  Savannas = 9,
  Grasslands = 10,
  PermanentWetlands = 11,
  Croplands = 12,
  UrbanBuilt = 13,
  CroplandMosaic = 14,
  SnowIce = 15,
  Barren = 16,
  Water = 17
}


export interface IWindProps {
  // Wind speed in mph.
  speed: number;
  // Angle in degrees following this definition: https://en.wikipedia.org/wiki/Wind_direction
  // 0 is northern wind, 90 is eastern wind.
  direction: number;
}
