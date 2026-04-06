import { Vegetation, TerrainType, DroughtLevel, VegetationType } from "../types";
import { observable, makeObservable } from "mobx";

export interface ZoneOptions {
  vegetation?: VegetationType;
  terrainType?: TerrainType;
  droughtLevel?: number;
}

// values for each level of vegetation: Grass, Shrub, Forest, ForestWithSuppression
export const moistureLookups: {[key in DroughtLevel]: number[]} = {
  [DroughtLevel.NoDrought]: [0.1275, 0.255, 0.17, 0.2125],
  [DroughtLevel.MildDrought]: [0.09, 0.18, 0.12, 0.15],
  [DroughtLevel.MediumDrought]: [0.0525, 0.105, 0.07, 0.0875],
  [DroughtLevel.SevereDrought]: [0.015, 0.03, 0.02, 0.025],
};

export const moistureLookupByLandCover: { [key in VegetationType]: number[] } = {
  [VegetationType.EvergreenNeedleleaf]: [0.25, 0.27, 0.22, 0.26],
  [VegetationType.EvergreenBroadleaf]: [0.28, 0.30, 0.25, 0.29],
  [VegetationType.DeciduousNeedleleaf]: [0.20, 0.24, 0.18, 0.21],
  [VegetationType.DeciduousBroadleaf]: [0.22, 0.26, 0.20, 0.23],
  [VegetationType.MixedForest]: [0.19, 0.23, 0.17, 0.21],
  [VegetationType.ClosedShrublands]: [0.14, 0.18, 0.12, 0.15],
  [VegetationType.OpenShrublands]: [0.10, 0.14, 0.08, 0.11],
  [VegetationType.WoodySavannas]: [0.12, 0.16, 0.10, 0.13],
  [VegetationType.Savannas]: [0.09, 0.13, 0.08, 0.10],
  [VegetationType.Grasslands]: [0.08, 0.12, 0.07, 0.09],
  [VegetationType.PermanentWetlands]: [0.30, 0.33, 0.28, 0.31],
  [VegetationType.Croplands]: [0.13, 0.16, 0.12, 0.15],
  [VegetationType.UrbanBuilt]: [0.05, 0.08, 0.04, 0.06],
  [VegetationType.CroplandMosaic]: [0.12, 0.15, 0.10, 0.13],
  [VegetationType.SnowIce]: [0.00, 0.01, 0.00, 0.00],
  [VegetationType.Barren]: [0.01, 0.02, 0.01, 0.01],
  [VegetationType.Water]: [0.00, 0.00, 0.00, 0.00]
};

export class Zone {
  @observable public vegetation: VegetationType = VegetationType.Barren;
  @observable public terrainType: TerrainType = TerrainType.Plains;
  @observable public droughtLevel: DroughtLevel = DroughtLevel.MildDrought;

  constructor(props?: ZoneOptions) {
    makeObservable(this);
    Object.assign(this, props);
  }

  clone() {
    return new Zone({
      vegetation: this.vegetation,
      terrainType: this.terrainType,
      droughtLevel: this.droughtLevel,
    });
  }
}
