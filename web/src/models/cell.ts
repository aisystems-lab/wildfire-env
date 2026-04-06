import { Zone, moistureLookupByLandCover, moistureLookups } from "./zone";
import { Vegetation, DroughtLevel, VegetationType } from "../types";

export enum FireState {
  Unburnt = 0,
  Burning = 1,
  Burnt = 2
}

// See: https://www.pivotaltracker.com/story/show/170344417
export enum BurnIndex {
  Low = 0,
  Medium = 1,
  High = 2
}

export interface CellOptions {
  x: number;
  y: number;
  zone: Zone;
  zoneIdx?: number;
  baseElevation?: number;
  ignitionTime?: number;
  fireState?: FireState;
  isUnburntIsland?: boolean;
  isRiver?: boolean;
  isFireLine?: boolean;
  isFireLineUnderConstruction?: boolean;
}

const FIRE_LINE_DEPTH = 2000;
const MAX_BURN_TIME = 500;

export class Cell {
  public x: number; // grid X coord
  public y: number; // grid Y coord
  public zone: Zone;
  public zoneIdx: number;
  public baseElevation = 0;
  public ignitionTime = Infinity;
  public spreadRate = 0;
  public burnTime: number = MAX_BURN_TIME;
  public fireState: FireState = FireState.Unburnt;
  public isUnburntIsland = false;
  public isFireSurvivor = false;
  public isRiver = false;
  public isFireLine = false;
  public isFireLineUnderConstruction = false;
  public helitackDropCount = 0;

  constructor(props: CellOptions) {
    Object.assign(this, props);
  }

  public get vegetation() {
    return this.zone.vegetation;
  }

  public get elevation() {
    if (this.isFireLine) {
      return this.baseElevation - FIRE_LINE_DEPTH;
    }
    return this.baseElevation;
  }

  public get isNonburnable() {
    return this.isRiver || this.isUnburntIsland;
  }

  public get moistureContent() {
    if (this.isNonburnable) {
      return Infinity;
    }
    return moistureLookupByLandCover[this.vegetation][this.droughtLevel];
  }

  public get droughtLevel() {
    if (this.helitackDropCount > 0) {
      const newDroughtLevel = this.zone.droughtLevel - this.helitackDropCount;
      return Math.max(newDroughtLevel, DroughtLevel.NoDrought) as DroughtLevel;
    }
    return this.zone.droughtLevel;
  }

  public get isBurningOrWillBurn() {
    return this.fireState === FireState.Burning || this.fireState === FireState.Unburnt && this.ignitionTime < Infinity;
  }

  public get canSurviveFire() {
    return this.burnIndex === BurnIndex.Low && this.vegetation === VegetationType.PermanentWetlands; //Accomodate all non-burnable vegetation types
  }

  // public get burnIndex() {
  //   // Values based on: https://www.pivotaltracker.com/story/show/170344417/comments/209774367
  //   if (this.vegetation === Vegetation.Grass) {
  //     if (this.spreadRate < 45) {
  //       return BurnIndex.Low;
  //     }
  //     return BurnIndex.Medium;
  //   }
  //   if (this.vegetation === Vegetation.Shrub) {
  //     if (this.spreadRate < 10) {
  //       return BurnIndex.Low;
  //     }
  //     if (this.spreadRate < 50) {
  //       return BurnIndex.Medium;
  //     }
  //     return BurnIndex.High;
  //   }
  //   if (this.vegetation === Vegetation.Forest) {
  //     if (this.spreadRate < 25) {
  //       return BurnIndex.Low;
  //     }
  //     return BurnIndex.Medium;
  //   }
  //   // this.vegetation === Vegetation.ForestWithSuppression
  //   if (this.spreadRate < 12) {
  //     return BurnIndex.Low;
  //   }
  //   if (this.spreadRate < 40) {
  //     return BurnIndex.Medium;
  //   }
  //   return BurnIndex.High;
  // }

    public get burnIndex(): BurnIndex {
    switch (this.vegetation) {
      case VegetationType.Grasslands:
        return this.spreadRate < 45 ? BurnIndex.Low : BurnIndex.Medium;

      case VegetationType.ClosedShrublands:
      case VegetationType.OpenShrublands:
        if (this.spreadRate < 10) return BurnIndex.Low;
        if (this.spreadRate < 50) return BurnIndex.Medium;
        return BurnIndex.High;

      case VegetationType.EvergreenNeedleleaf:
      case VegetationType.DeciduousNeedleleaf:
      case VegetationType.MixedForest:
      case VegetationType.EvergreenBroadleaf:
      case VegetationType.DeciduousBroadleaf:
        return this.spreadRate < 25 ? BurnIndex.Low : BurnIndex.Medium;

      default:
        // other vegetation types default to low burn index
        return BurnIndex.Low;
    }
  }

  public isBurnableForBI(burnIndex: BurnIndex) {
    // Fire lines will burn when burn index is high.
    return !this.isNonburnable && (!this.isFireLine || burnIndex === BurnIndex.High);
  }

  public reset() {
    this.ignitionTime = Infinity;
    this.spreadRate = 0;
    this.burnTime = MAX_BURN_TIME;
    this.fireState = FireState.Unburnt;
    this.isFireLineUnderConstruction = false;
    this.isFireLine = false;
    this.helitackDropCount = 0;
    this.isFireSurvivor = false;
  }
}
