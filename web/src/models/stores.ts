import { SimulationModel } from "./simulation";
import { UIModel } from "./ui";
import presets from "../presets";
import { getDefaultConfig, getUrlConfig } from "../config";
import { DroughtLevel, TerrainType, Vegetation } from "../types";

export interface IStores {
  simulation: SimulationModel;
  ui: UIModel;
}

export const createStores = (): IStores => {
  const simulation = new SimulationModel(
    presets[getUrlConfig().preset || getDefaultConfig().preset],
    false
  );
  (window as any).sim = simulation;
  (window as any).DroughtLevel = DroughtLevel;
  (window as any).Vegetation = Vegetation;
  (window as any).TerrainType = TerrainType;
  return {
    simulation,
    ui: new UIModel(),
  };
};
