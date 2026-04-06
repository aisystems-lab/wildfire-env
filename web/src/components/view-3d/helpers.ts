import { SimulationModel } from "../../models/simulation";

export const DEFAULT_UP: [number, number, number] = [0, 0, 1];

export const PLANE_WIDTH = 1;

export const planeHeight = (simulation: SimulationModel) =>
  simulation.config.modelHeight * PLANE_WIDTH / simulation.config.modelWidth;

// Ratio between model unit (feet) and 3D view distance unit (unitless).
export const ftToViewUnit = (simulation: SimulationModel) => PLANE_WIDTH / simulation.config.modelWidth;

