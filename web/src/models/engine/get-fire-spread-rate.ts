import { Fuel , Vegetation, IWindProps, VegetationType } from "../../types";
import { Vector2 } from "three";

interface ICellProps {
  x: number;
  y: number;
  vegetation: Vegetation;
  moistureContent: number;
  elevation: number;
}

const heatContent = 8000;
const totalMineralContent = 0.0555;
const effectiveMineralContent = 0.01;

/**
 * Rothermel‐style fuel parameters for each VegetationType.
 * Grass, Shrub and Forest come straight from your PT story.
 * Others are interpolated or approximated so that 
 * packingRatio, netFuelLoad and sav decrease/increase sensibly.
 */

const FuelConstants: { [key in VegetationType]: Fuel } = {
  // ----- True surface fuels -----
  [VegetationType.Grasslands]: {
    // From PT: fine grass
    sav: 2100,            // very fine
    netFuelLoad: 0.294,   // kg/m²
    fuelBedDepth: 3.0,    // m
    packingRatio: 0.00306,
    mx: 0.15
  },
  [VegetationType.ClosedShrublands]: {
    // From PT: shrub
    sav: 1672,
    netFuelLoad: 0.239,
    fuelBedDepth: 1.2,
    packingRatio: 0.01198,
    mx: 0.30
  },
  [VegetationType.OpenShrublands]: {
    // slightly coarser / less fuel than closed shrub
    sav: 1500,
    netFuelLoad: 0.20,
    fuelBedDepth: 1.0,
    packingRatio: 0.010,
    mx: 0.30
  },

  // ----- Forest litter (very fine needles) -----
  [VegetationType.EvergreenNeedleleaf]: {
    // approximate “Forest” from PT
    sav: 1716,
    netFuelLoad: 0.0459,
    fuelBedDepth: 0.10,
    packingRatio: 0.04878,
    mx: 0.20
  },
  [VegetationType.DeciduousNeedleleaf]: {
    // similar to evergreen needles
    sav: 1650,
    netFuelLoad: 0.040,
    fuelBedDepth: 0.10,
    packingRatio: 0.045,
    mx: 0.20
  },

  [VegetationType.EvergreenBroadleaf]: {
    // leaf litter is a bit broader but still light
    sav: 1500,
    netFuelLoad: 0.06,
    fuelBedDepth: 0.12,
    packingRatio: 0.035,
    mx: 0.25
  },
  [VegetationType.DeciduousBroadleaf]: {
    sav: 1400,
    netFuelLoad: 0.05,
    fuelBedDepth: 0.10,
    packingRatio: 0.030,
    mx: 0.25
  },

  [VegetationType.MixedForest]: {
    // blend of needle + broadleaf
    sav: (1716 + 1500) / 2,    // ~1608
    netFuelLoad: (0.0459 + 0.06) / 2,  // ~0.053
    fuelBedDepth: 0.11,
    packingRatio: 0.042,
    mx: 0.22
  },

  // ----- Savanna / woody grassland -----
  [VegetationType.WoodySavannas]: {
    sav: 1400,
    netFuelLoad: 0.18,
    fuelBedDepth: 0.8,
    packingRatio: 0.008,
    mx: 0.30
  },
  [VegetationType.Savannas]: {
    sav: 1450,
    netFuelLoad: 0.17,
    fuelBedDepth: 0.7,
    packingRatio: 0.0085,
    mx: 0.25
  },

  // ----- Agricultural -----
  [VegetationType.Croplands]: {
    // crop residue & stubble (moderate fine fuel)
    sav: 1200,
    netFuelLoad: 0.20,
    fuelBedDepth: 0.50,
    packingRatio: 0.015,
    mx: 0.30
  },
  [VegetationType.CroplandMosaic]: {
    // mix of crop + natural
    sav: 1300,
    netFuelLoad: 0.18,
    fuelBedDepth: 0.60,
    packingRatio: 0.012,
    mx: 0.25
  },

  // ----- Wet or sparsely vegetated -----
  [VegetationType.PermanentWetlands]: {
    // mostly saturated ground, little burnable
    sav: 800,
    netFuelLoad: 0.12,
    fuelBedDepth: 0.50,
    packingRatio: 0.007,
    mx: 0.50
  },
  [VegetationType.Barren]: {
    // sparse grass / bare
    sav: 600,
    netFuelLoad: 0.01,
    fuelBedDepth: 0.05,
    packingRatio: 0.002,
    mx: 0.10
  },

  // ----- Non‐burnable -----
  [VegetationType.UrbanBuilt]: {
    sav: 0, netFuelLoad: 0, fuelBedDepth: 0, packingRatio: 0, mx: 0
  },
  [VegetationType.SnowIce]: {
    sav: 0, netFuelLoad: 0, fuelBedDepth: 0, packingRatio: 0, mx: 0
  },
  [VegetationType.Water]: {
    sav: 0, netFuelLoad: 0, fuelBedDepth: 0, packingRatio: 0, mx: 0
  }
};



// Values are specified in this PT story: https://www.pivotaltracker.com/story/show/170343321
// const FuelConstants: {[key in Vegetation]: Fuel} = {
//   [Vegetation.Grass]: {
//     sav: 2100,
//     netFuelLoad: 0.294,
//     fuelBedDepth: 3,
//     packingRatio: 0.00306,
//     mx: 0.15
//   },
//   [Vegetation.Shrub]: {
//     sav: 1672,
//     netFuelLoad: 0.239,
//     fuelBedDepth: 1.2,
//     packingRatio: 0.01198,
//     mx: 0.3
//   },
//   // TODO: the following two land types have not yet been configured via specification,
//   // only by approximation to get the code to compile
//   [Vegetation.Forest]: {
//     sav: 1716,
//     netFuelLoad: 0.0459,
//     fuelBedDepth: 0.1,
//     packingRatio: 0.04878,
//     mx: 0.2
//   },
//   [Vegetation.ForestWithSuppression]: {
//     sav: 1500,
//     netFuelLoad: 0.689,
//     fuelBedDepth: 0.5,
//     packingRatio: 0.02224,
//     mx: 0.25
//   }
// };

// Helper vector used repeatedly in other calculations.
const ORIGIN = new Vector2(0, 0);

const dist = (c1: ICellProps, c2: ICellProps) => {
  const xDiff = c1.x - c2.x;
  const yDiff = c1.y - c2.y;
  if (xDiff === 0 || yDiff === 0) {
    return Math.abs(xDiff + yDiff);
  } else {
    return Math.sqrt(xDiff * xDiff + yDiff * yDiff);
  }
};

/**
 * The mathematical model is derived from
 *   https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf
 * section "6.2 Fire Spread from a Single Ignition Point", page 88.
 *
 * Returns multiplier that should be applied to the max fire spread rate to calculate final value in given direction.
 *
 * @param maxSpreadDirection angle from positive X axis
 */
export const getDirectionFactor =
  (sourceCell: ICellProps, targetCell: ICellProps, effectiveWindSpeed: number, maxSpreadDirection: number) => {
  // Note that wind speed in our model is usually defined in feet/min. However, this formula is using miles per hour.
  const effectiveWindSpeedMPH = effectiveWindSpeed / 88;
  const Z = 1 + 0.25 * effectiveWindSpeedMPH;
  const e = Math.pow(Math.pow(Z, 2) - 1, 0.5) / Z;

  const cellCentersVector = new Vector2(targetCell.x - sourceCell.x, targetCell.y - sourceCell.y);
  // Angle between cells centers and direction of max fire spread.
  const relativeAngle = Math.abs(cellCentersVector.angle() - maxSpreadDirection);

  return (1 - e) / (1 - e * Math.cos(relativeAngle));
};

/**
 * The mathematical fire model is derived from
 *   https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf
 * and was made into a spreadsheet that can be seen at
 *   https://docs.google.com/spreadsheets/d/1ov3JUz6hXdnXChbXTz20Fo_9YoWmGukJgCaIMJeRUb4/
 *   UPDATED version: https://drive.google.com/file/d/1ck0nwlawOtK-GjCV4qJ6ztMcxh3utbv-/view
 *
 * Still to do:
 *  * Calculate the slope between two cells
 *  * Scale the value given the angle between these two cells and the wind
 *  * Break up this function into several curried functions which will allow us to calculate
 *    the spread times between two cells as quickly as possible. (Fuel types and magnitude of
 *    the wind can be curried, angle and slope will be calculated per pair)
 *
 * @param sourceCell Grid cell that is currently BURNING
 * @param targetCell Adjacent grid cell that is currently UNBURNT
 * @param wind Wind properties, speed and direction
 * @param cellSize cell size in feet
 *
 * @return fire spread rate in ft/min
 */
export const getFireSpreadRate = (
  sourceCell: ICellProps,
  targetCell: ICellProps,
  wind: IWindProps,
  cellSize: number
) => {
  const fuel = FuelConstants[targetCell.zone.vegetation];
  console.log(fuel);
  
  const sav = fuel.sav;
  const packingRatio = fuel.packingRatio;
  const netFuelLoad = fuel.netFuelLoad;
  const mx = fuel.mx;
  const fuelBedDepth = fuel.fuelBedDepth;

  const moistureContentRatio = targetCell.moistureContent / mx;
  const savFactor = Math.pow(sav, 1.5);

  const a = 133 * Math.pow(sav, -0.7913);
  const b = 0.02526 * Math.pow(sav, 0.54);
  const c = 7.47 * Math.exp(-0.133 * Math.pow(sav, 0.55));
  const e = 0.715 * (-0.000359 * sav);

  const maximumReactionVelocity = savFactor * Math.pow(495 + (0.0594 * savFactor), -1);
  const optimumPackingRatio = 3.348 * Math.pow(sav, -0.8189);
  const optimumReactionVelocity = maximumReactionVelocity * Math.pow(packingRatio / optimumPackingRatio, a)
          * Math.exp(a * (1 - (packingRatio / optimumPackingRatio)));
  const moistureDampingCoefficient = 1 - (2.59 * moistureContentRatio) + (5.11 * Math.pow(moistureContentRatio, 2))
          - (3.52 * Math.pow(moistureContentRatio, 3));
  const mineralDampingCoefficient = 0.174 * Math.pow(effectiveMineralContent, -0.19);
  const reactionIntensity = optimumReactionVelocity * netFuelLoad * heatContent
          * moistureDampingCoefficient * mineralDampingCoefficient;

  const propagatingFluxRatio = Math.pow(192 + (0.2595 * sav), -1)
          * Math.exp((0.792 + (0.681 * Math.pow(sav, 0.5)) ) * (packingRatio + 0.1));

  const fuelLoad = netFuelLoad / (1 - totalMineralContent);
  const ovenDryBulkDensity = fuelLoad / fuelBedDepth;

  const effectiveHeatingNumber = Math.exp(-138 / sav);

  const heatOfPreIgnition = 250 + (1116 * targetCell.moistureContent);

  // r0 is rate of spread without considering wind and slope.
  const r0 = reactionIntensity * propagatingFluxRatio /
             (ovenDryBulkDensity * effectiveHeatingNumber * heatOfPreIgnition);

  const windSpeedFtPerMin = wind.speed * 88;
  const windFactor = c * Math.pow(windSpeedFtPerMin, b) * Math.pow((packingRatio / optimumPackingRatio), -e);

  const distInFt = dist(sourceCell, targetCell) * cellSize;
  const elevationDiffInFt = targetCell.elevation - sourceCell.elevation;
  const slopeTan = elevationDiffInFt / distInFt;
  const slopeFactor = 5.275 * Math.pow(packingRatio, -0.3) * Math.pow(slopeTan, 2);

  // Now, follow calculations from section "6.1 Fire Behavior in the Direction of Maximum Spread", p. 85.
  // Note that "t" is assumed to be 1 to simplify calculations. Note that instead of assuming that slope vector
  // is (ds, 0) and calculating angle between upslope vector and wind vector, we just use real angles, but set lengths
  // correctly and use vector addition. It makes calculations more concise.

  // 0 degrees is northern wind, so wind vector is pointing down (south). 90 deg should be eastern wind.
  const windVector = (new Vector2(0, -1)).rotateAround(ORIGIN, -wind.direction * Math.PI / 180);
  const upslopeVector = targetCell.elevation >= sourceCell.elevation ?
          new Vector2(targetCell.x - sourceCell.x, targetCell.y - sourceCell.y) :
          new Vector2(sourceCell.x - targetCell.x, sourceCell.y - targetCell.y);

  const dw = r0 * windFactor; // * t (but t = 1)
  windVector.setLength(dw);

  const ds = r0 * slopeFactor; // * t (but t = 1)
  upslopeVector.setLength(ds);

  const maxSpreadRateVector = (new Vector2()).addVectors(upslopeVector, windVector);

  // rh is max spread rate that already includes wind and slope factors. See table 26, page 86.
  // Note that dh = maxSpreadRateVector.length();
  const rh = r0 + maxSpreadRateVector.length();

  // Effective wind factor can be calculated from rh and r0. It's an abstract value that includes
  // both wind and slope effects. See table 26, page 86.
  const effectiveWindFactor = rh / r0 - 1;

  // Effective wind speed is calculated using inverted wind factor equation where we use effective wind factor.
  // It's an abstract value that includes both wind and slope effects.
  const effectiveWindSpeed = Math.pow(effectiveWindFactor /
          (c * Math.pow(packingRatio / optimumPackingRatio, -e)), 1 / b);

  const directionFactor = getDirectionFactor(sourceCell, targetCell, effectiveWindSpeed, maxSpreadRateVector.angle());
  return rh * directionFactor;
};
