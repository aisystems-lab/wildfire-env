import { Vector2 } from "three";
import { BurnIndex, Cell, FireState } from "../cell";
import { getFireSpreadRate } from "./get-fire-spread-rate";
import { IWindProps } from "../../types";
import { dist, withinDist, getGridIndexForLocation, forEachPointBetween, directNeighbours, } from "../utils/grid-utils";

const modelDay = 1440; // minutes

// const endOfLowIntensityFireProbability: {[key: number]: number} = {
//   0: 0.0,
//   1: 0.6,
//   2: 0.6,
//   3: 0.7,
//   4: 0.8,
//   5: 1.0
// };

// Lowered probabilities so the fire persists much longer:
const endOfLowIntensityFireProbability: { [day: number]: number } = {
  0: 0.0,   // cannot end on day 0
  1: 0.1,   // only 10% chance to stop on day 1
  2: 0.1,   // 10% on day 2
  3: 0.2,   // 20% on day 3
  4: 0.3,   // 30% on day 4
  5: 0.5,   // 50% on day 5
  6: 0.7,   // 70% on day 6
  7: 1.0    // guaranteed to end on day 7
};

export const nonburnableCellBetween = (
  cells: Cell[], width: number, x0: number, y0: number, x1: number, y1: number, burnIndex: BurnIndex
) => {
  let result = false;
  forEachPointBetween(x0, y0, x1, y1, (x: number, y: number) => {
    const idx = getGridIndexForLocation(x, y, width);
    if (!cells[idx].isBurnableForBI(burnIndex)) {
      result = true;
    }
  });
  return result;
};

/**
 * Returns an array of indices of all cells neighboring `i`.
 * Each cell within `neighborsDist` is considered to be a neighbour if there's no river or fire line between
 * this cell and cell `i`.
 */
export const getGridCellNeighbors = (
  cells: Cell[], i: number, width: number, height: number, neighborsDist: number, burnIndex: BurnIndex
) => {
  const neighbours: number[] = [];
  const queue: number[] = [];
  const processed: {[key: number]: boolean}  = {};
  const x0 = i % width;
  const y0 = Math.floor(i / width);
  // Keep this flag for performance reasons. If there's no nonburnable ceels in current grid area, it doesn't
  // make sense to run Bresenham's algorithm for every cell (nonburnableCellBetween).
  let anyNonburnableCells = false;
  // Start BFS.
  queue.push(i);
  processed[i] = true;
  while (queue.length > 0) {
    const j = queue.shift() as number;
    const x1 = j % width;
    const y1 = Math.floor(j / width);
    directNeighbours.forEach(diff => {
      const nIdx = getGridIndexForLocation(x1 + diff.x, y1 + diff.y, width);
      if (x1 + diff.x >= 0 && x1 + diff.x < width && y1 + diff.y >= 0 &&  y1 + diff.y < height &&
        !processed[nIdx] &&
        withinDist(x0, y0, x1 + diff.x, y1 + diff.y, neighborsDist)
      ) {
        if (!cells[nIdx].isBurnableForBI(burnIndex)) {
          anyNonburnableCells = true;
        } else if (!anyNonburnableCells || !nonburnableCellBetween(cells, width, x1 + diff.x, y1 + diff.y, x0, y0, burnIndex)) {
          neighbours.push(nIdx);
          queue.push(nIdx);
        }
        processed[nIdx] = true;
      }
    });
  }
  return neighbours;
};

export interface IFireEngineConfig {
  gridWidth: number;
  gridHeight: number;
  cellSize: number;
  minCellBurnTime: number;
  neighborsDist: number;
  fireSurvivalProbability: number;
}

// Lightweight helper that is responsible only for math calculations. It's not bound to MobX or any UI state
// (it's role of the Simulation model). Config properties are explicitly listed, so it's clear
// which config options are responsible for simulation progress.
export class FireEngine {
  public cells: Cell[];
  public wind: IWindProps;
  public gridWidth: number;
  public gridHeight: number;
  public cellSize: number;
  public minCellBurnTime: number;
  public neighborsDist: number;
  public fireSurvivalProbability: number;
  public endOfLowIntensityFire = false;
  public fireDidStop = false;
  public day = 0;
  public burnedCellsInZone: {[key: number]: number} = {};

  constructor(cells: Cell[], wind: IWindProps, sparks: Vector2[], config: IFireEngineConfig) {
    this.cells = cells;
    this.wind = wind;
    this.gridWidth = config.gridWidth;
    this.gridHeight = config.gridHeight;
    this.cellSize = config.cellSize;
    this.minCellBurnTime = config.minCellBurnTime;
    this.neighborsDist = config.neighborsDist;
    this.fireSurvivalProbability = config.fireSurvivalProbability;

    // Use sparks to start the simulation.
    sparks.forEach(spark => {
      const sparkCell = this.cellAt(spark.x, spark.y);
      sparkCell.ignitionTime = 0;
      if (sparkCell.isUnburntIsland) {
        // If spark is placed inside unburnt island, remove this island as otherwise the fire won't pick up.
        this.removeUnburntIsland(sparkCell);
      }
    });
  }

  public cellAt(x: number, y: number) {
    const gridX = Math.floor(x / this.cellSize);
    const gridY = Math.floor(y / this.cellSize);
    return this.cells[getGridIndexForLocation(gridX, gridY, this.gridWidth)];
  }

  public removeUnburntIsland(startingCell: Cell) {
    const queue: Cell[] = [];
    startingCell.isUnburntIsland = false;
    queue.push(startingCell);
    while (queue.length > 0) {
      const c = queue.shift() as Cell;
      directNeighbours.forEach(diff => {
        const x1 = c.x + diff.x;
        const y1 = c.y + diff.y;
        const nIdx = getGridIndexForLocation(x1, y1, this.gridWidth);
        if (x1 >= 0 && x1 < this.gridWidth && y1 >= 0 && y1 < this.gridHeight && this.cells[nIdx].isUnburntIsland) {
          this.cells[nIdx].isUnburntIsland = false;
          queue.push(this.cells[nIdx]);
        }
      });
    }
  }

  public updateFire(time: number) {
    // Each time a day changes check if the low intensity fire shouldn't go out on itself.
    const newDay = Math.floor(time / modelDay);
    if (newDay !== this.day) {
      this.day = newDay;
      if (Math.random() <= (endOfLowIntensityFireProbability[newDay] ?? 0.0)) {
        this.endOfLowIntensityFire = true;
      }
    }

    const numCells = this.cells.length;
    // Run through all cells. Check the unburnt neighbors of currently-burning cells. If the current time
    // is greater than the ignition time of the cell and the delta time for the neighbor, update
    // the neighbor's ignition time.
    // At the same time, we update the unburnt/burning/burnt states of the cells.
    const newIgnitionData: number[] = [];
    const newFireStateData: FireState[] = [];
    // Start with assumption that fire stopped.
    // debugger;
    this.fireDidStop = true;

    for (let i = 0; i < numCells; i++) {
      const cell = this.cells[i];
      if (cell.isBurningOrWillBurn) {
        this.fireDidStop = false; // fire still going on
        // debugger;
      }
      const ignitionTime = cell.ignitionTime;
      if (cell.fireState === FireState.Burning && time - ignitionTime > cell.burnTime) {
        newFireStateData[i] = FireState.Burnt;
        if (cell.canSurviveFire && Math.random() < this.fireSurvivalProbability) {
          cell.isFireSurvivor = true;
        }
      } else if (cell.fireState === FireState.Unburnt && time > ignitionTime ) {
        // Sets any unburnt cells to burning if we are passed their ignition time.
        // Although during a simulation all cells will have their state sent to BURNING through the process
        // above, this not only allows us to pre-set ignition times for testing, but will also allow us to
        // run forward or backward through a simulation.
        newFireStateData[i] = FireState.Burning;
        if (!this.burnedCellsInZone[cell.zoneIdx]) {
          this.burnedCellsInZone[cell.zoneIdx] = 1;
        } else {
          this.burnedCellsInZone[cell.zoneIdx] += 1;
        }
        // Fire should spread if endOfLowIntensityFire flag is false or burn index is high enough.
        const fireShouldSpread = !this.endOfLowIntensityFire || cell.burnIndex !== BurnIndex.Low;
        if (fireShouldSpread) {
          // Fire lines and other fire control methods will work only if burn index is low or medium.
          // If it's high, fire cannot be controlled.
          const neighbors = getGridCellNeighbors(this.cells, i, this.gridWidth, this.gridHeight, this.neighborsDist, cell.burnIndex);
          neighbors.forEach(n => {
            
            const neighCell = this.cells[n];
            const distInFt = dist(cell.x, cell.y, neighCell.x, neighCell.y) * this.cellSize;
            const spreadRate = getFireSpreadRate(cell, neighCell, this.wind, this.cellSize);
            const spreadRateIncDistance = spreadRate / distInFt;
            const ignitionDelta = 1 / spreadRateIncDistance;
            if (neighCell.fireState === FireState.Unburnt) {
              newIgnitionData[n] = Math.min(
                ignitionTime + ignitionDelta, newIgnitionData[n] || neighCell.ignitionTime
              );
              // Make cell burn time proportional to fire spread rate.
              const newBurnTime = (newIgnitionData[n] - ignitionTime) + this.minCellBurnTime;
              if (newBurnTime < neighCell.burnTime) {
                neighCell.burnTime = newBurnTime;
              }
              // Save max spread rate.
              if (spreadRate > neighCell.spreadRate) {
                neighCell.spreadRate = spreadRate;
              }
            }
          });
        }
      }
    }
    // debugger;
    for (let i = 0; i < numCells; i++) {
      if (newFireStateData[i] !== undefined) {
        this.cells[i].fireState = newFireStateData[i];
      }
      if (newIgnitionData[i] !== undefined) {
        this.cells[i].ignitionTime = newIgnitionData[i];
      }
    }
  }
}
