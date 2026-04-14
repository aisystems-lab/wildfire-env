import { action, computed, observable, makeObservable } from "mobx";
import { IWindProps, TerrainType } from "../types";
import {  BurnIndex, Cell, CellOptions, FireState } from "./cell";
import { getDefaultConfig, ISimulationConfig, getUrlConfig } from "../config";
import { Vector2 } from "three";
import { getElevationData, getLandCoverZoneIndex, getRiverData, getUnburntIslandsData, getZoneIndex } from "./utils/data-loaders";
import { Zone } from "./zone";
import { FireEngine } from "./engine/fire-engine";
import { getGridIndexForLocation, forEachPointBetween, dist } from "./utils/grid-utils";
import { WS_URL } from "../env";

interface ICoords {
  x: number;
  y: number;
}

// When config.changeWindOnDay is defined, but config.newWindSpeed is not, the model will use random value limited
// by this constant.
const NEW_WIND_MAX_SPEED = 20; // mph

const DEFAULT_ZONE_DIVISION = {
  2: [
    [0, 1]
  ],
  3: [
    [0, 1, 2],
  ]
};

// This class is responsible for data loading, adding sparks and fire lines and so on. It's more focused
// on management and interactions handling. Core calculations are delegated to FireEngine.
// Also, all the observable properties should be here, so the view code can observe them.
export class SimulationModel {
  public config: ISimulationConfig;
  public prevTickTime: number | null;
  public dataReadyPromise: Promise<void>;
  public engine: FireEngine | null = null;
  public zoneIndex: number[][] | string = [];
  // Cells are not directly observable. Changes are broadcasted using cellsStateFlag and cellsElevationFlag.
  public cells: Cell[] = [];

  public userDefinedWind: IWindProps | undefined = undefined;
  // This property is also used by the UI to highlight wind info box.
  @observable public windDidChange = false;
  @observable public backendDriven = false;

  @observable public time = 0;
  @observable public dataReady = false;
  @observable public wind: IWindProps;
  @observable public sparks: Vector2[] = [];
  @observable public fireLineMarkers: Vector2[] = [];
  @observable public zones: Zone[] = [];
  @observable public simulationStarted = false;
  @observable public simulationRunning = false;
  @observable public gymAllowedContinue = true;
  @observable public lastFireLineTimestamp = -Infinity;
  @observable public lastHelitackTimestamp = -Infinity;
  @observable public totalCellCountByZone: {[key: number]: number} = {};
  @observable public burnedCellsInZone: {[key: number]: number} = {};
  // These flags can be used by view to trigger appropriate rendering. Theoretically, view could/should check
  // every single cell and re-render when it detects some changes. In practice, we perform these updates in very
  // specific moments and usually for all the cells, so this approach can be way more efficient.
  @observable public cellsStateFlag = 0;
  @observable public cellsElevationFlag = 0;
  @observable public episodeCount = 0;
  @observable public stepCount = 0;
  @observable public cellsBurning = 0;
  @observable public cellsBurnt = 0;
  @observable public helicopterGridCoord: [number, number] = [70, 30];
  @observable public helicopterModelPosition = new Vector2(0, 0);

 // Create a new WebSocket object
//  private socket = new WebSocket('ws://localhost:8765');
private socket: WebSocket | null = null;
private shouldReconnect = true;

// WebSocket Event Handlers
constructor(presetConfig: Partial<ISimulationConfig>, autoloadTerrain = true) {
    makeObservable(this);
    this.dataReadyPromise = Promise.resolve();
    this.configure(presetConfig);
    if (autoloadTerrain) {
      this.populateCellsData();
    }
}

private backendGridToModelFeet(gridX: number, gridY: number) {
  return {
    x: (gridX + 0.5) * this.config.cellSize,
    y: (this.gridHeight - gridY - 0.5) * this.config.cellSize
  };
}

private buildBackendZones() {
  return Array.from({ length: 18 }, (_, idx) => new Zone({
    terrainType: this.config.zones[idx]?.terrainType ?? TerrainType.Plains,
    vegetation: (idx + 1) as any,
    droughtLevel: this.config.zones[idx]?.droughtLevel ?? 0
  }));
}

@action
private initializeBackendTerrain(message: any) {
  const vegetationGrid = message.vegetation;
  const elevationGrid = message.elevation;
  const riverGrid = message.is_river;

  if (!Array.isArray(vegetationGrid) || !Array.isArray(elevationGrid) || !Array.isArray(riverGrid)) {
    console.warn("Invalid terrain payload", message);
    return;
  }

  if (typeof message.grid_width === "number") {
    this.config.gridWidth = message.grid_width;
  }
  if (typeof message.model_width === "number") {
    this.config.modelWidth = message.model_width;
  }
  if (typeof message.model_height === "number") {
    this.config.modelHeight = message.model_height;
  }
  if (typeof message.heightmap_max_elevation === "number") {
    this.config.heightmapMaxElevation = message.heightmap_max_elevation;
  }

  this.backendDriven = true;
  this.dataReady = false;
  this.engine = null;
  this.zones = this.buildBackendZones();
  this.cells.length = 0;
  this.totalCellCountByZone = {};
  this.sparks.length = 0;

  if (Array.isArray(message.spark_coord) && message.spark_coord.length === 2) {
    const [sparkX, sparkY] = message.spark_coord;
    const spark = this.backendGridToModelFeet(sparkX, sparkY);
    this.setSpark(0, spark.x, spark.y);
  }

  if (Array.isArray(message.helicopter_coord) && message.helicopter_coord.length === 2) {
    const [heliX, heliY] = message.helicopter_coord;
    const helicopter = this.backendGridToModelFeet(heliX, heliY);
    this.helicopterGridCoord = [heliX, heliY];
    this.helicopterModelPosition = new Vector2(helicopter.x, helicopter.y);
  }

  for (let y = 0; y < this.gridHeight; y++) {
    const vegetationRow = vegetationGrid[y];
    const elevationRow = elevationGrid[y];
    const riverRow = riverGrid[y];

    for (let x = 0; x < this.gridWidth; x++) {
      const vegetation = Array.isArray(vegetationRow) ? (vegetationRow[x] ?? 16) : 16;
      const zoneIdx = Math.max(0, Math.min(this.zones.length - 1, vegetation - 1));
      const cellOptions: CellOptions = {
        x,
        y,
        zone: this.zones[zoneIdx],
        zoneIdx,
        baseElevation: Array.isArray(elevationRow) ? (elevationRow[x] ?? 0) : 0,
        isRiver: Array.isArray(riverRow) ? Boolean(riverRow[x]) : false,
      };
      this.cells.push(new Cell(cellOptions));
      this.totalCellCountByZone[zoneIdx] = (this.totalCellCountByZone[zoneIdx] ?? 0) + 1;
    }
  }

  this.dataReady = true;
  this.dataReadyPromise = Promise.resolve();
  this.updateCellsElevationFlag();
  this.updateCellsStateFlag();
}

private async applyBackendSnapshot(message: any) {
  if (!this.dataReady) {
    await this.dataReadyPromise;
  }

  const fireStateGrid = message.fire_state;
  const helitackDropsGrid = message.helitack_drops;

  if (!Array.isArray(fireStateGrid) || !Array.isArray(helitackDropsGrid)) {
    console.warn("Invalid snapshot payload", message);
    return;
  }

  this.backendDriven = true;
  this.simulationStarted = true;
  this.simulationRunning = !(message.terminated || message.truncated);
  this.config.showBurnIndex = false;
  this.time = typeof message.simulation_time === "number" ? message.simulation_time : this.time;
  this.episodeCount = typeof message.episode === "number" ? message.episode : this.episodeCount;
  this.stepCount = typeof message.step_count === "number" ? message.step_count : this.stepCount;
  this.cellsBurning = typeof message.cells_burning === "number" ? message.cells_burning : this.cellsBurning;
  this.cellsBurnt = typeof message.cells_burnt === "number" ? message.cells_burnt : this.cellsBurnt;

  if (Array.isArray(message.helicopter_coord) && message.helicopter_coord.length === 2) {
    const [heliX, heliY] = message.helicopter_coord;
    const helicopter = this.backendGridToModelFeet(heliX, heliY);
    this.helicopterGridCoord = [heliX, heliY];
    this.helicopterModelPosition = new Vector2(helicopter.x, helicopter.y);
    console.debug("Helicopter snapshot", {
      grid: { x: heliX, y: heliY },
      modelFeet: { x: helicopter.x, y: helicopter.y }
    });
  }

  if (Array.isArray(message.spark_coord) && message.spark_coord.length === 2) {
    const [sparkX, sparkY] = message.spark_coord;
    this.sparks.length = 0;
    const spark = this.backendGridToModelFeet(sparkX, sparkY);
    this.setSpark(0, spark.x, spark.y);
  }

  for (let y = 0; y < this.gridHeight; y++) {
    const fireStateRow = fireStateGrid[y];
    const helitackRow = helitackDropsGrid[this.gridHeight - 1 - y];
    if (!Array.isArray(fireStateRow) || !Array.isArray(helitackRow)) {
      continue;
    }

    for (let x = 0; x < this.gridWidth; x++) {
      const index = getGridIndexForLocation(x, y, this.gridWidth);
      const cell = this.cells[index];
      if (!cell) {
        continue;
      }

      cell.fireState = fireStateRow[x] ?? FireState.Unburnt;
      cell.helitackDropCount = helitackRow[x] ?? 0;
    }
  }

  this.updateCellsStateFlag();
}

public connectSocket() {
  this.shouldReconnect = true;
  if (this.socket && this.socket.readyState === 1) {
    console.log("🔁 WebSocket already connected.");
    return;
  }
  // const host = window.location.hostname;
  // const wsUrl = `ws://${host}:8765`;
  // this.socket = new WebSocket(wsUrl);
  // this.socket = new WebSocket("ws://python-backend:8765");
  // const WS_URL = process.env.REACT_APP_WS_URL
  this.socket = new WebSocket(WS_URL);
  console.log("🌐 Connecting to WebSocket server...");

  this.socket.onopen = () => {
    // this.reload();
    // this.handleReset();
    console.log("✅ Connected to the WebSocket server");
    console.log('Socket', this.socket);

    console.log(this.gridHeight,this.gridWidth)
    
    
    this.socket?.send(JSON.stringify({ type: "hello", role: "renderer" }));

    // const location = useLocation();
    const params = new URLSearchParams(window.location.search);

    const lat = params.get('lat');
    const lon = params.get('lon');
    const date = params.get('date');

    console.log({ lat, lon, date });
  };

  this.socket.onmessage = (event) => {
    console.log("🔥 Received from Python:", event.data);
    const message = JSON.parse(event.data);
    if (message.type === "terrain_init") {
      this.initializeBackendTerrain(message);
    } else if (message.type === "snapshot") {
      void this.applyBackendSnapshot(message);
    } else if (message.type === "pong") {
      console.log("✅ Received pong from server");
    }

    this.gymAllowedContinue = true;
    requestAnimationFrame(this.rafCallback);
  };

  this.socket.onerror = (err) => {
    console.error("❌ WebSocket error:", err);
  };

  this.socket.onclose = (e) => {
    if (!this.shouldReconnect) {
      return;
    }
    console.warn("⚠️ WebSocket closed. Attempting reconnect in 2s...");
    setTimeout(() => this.connectSocket(), 2000);
  };
}

  

  // Reset handler when receiving 'reset' action
  @action
  private async handleReset() {
    console.log('HANDLE RESET IS GETTING CALLED');
    
    this.reload();

    await this.dataReadyPromise;
    this.setSpark(0, 60000 - 1, 40000 - 1);

    this.start();
    const cells2D = this.generateFireStatusMapFromCells(
      this.engine?.cells ?? [],
      this.gridWidth,
      this.gridHeight
    );

    const response = {
      cells: cells2D,
      done: false,
      cellsBurning: 0,
      cellsBurnt: 0,
      quenchedCells: 0,
      on_fire: false
    };

    if (this.socket?.readyState === WebSocket.OPEN) {
      console.log("🧩 React socket object:", this.socket);

      await new Promise(res => setTimeout(res, 100));
      this.socket?.send(JSON.stringify(response));

      console.log("✅ Reset response sent");
    } else {
      console.warn("❌ Socket not open on reset");
    }
  }

  @action
  private handleHelicopterMovement(helicopter_coord: number[]) {
      if (helicopter_coord) {
        // Use coordinates directly - Python already sends them as [x, y]
        let array_x = helicopter_coord[0];  // First element is x
        let array_y = helicopter_coord[1];  // Second element is y
  
        // Ensure coordinates are within bounds
        array_x = Math.max(0, Math.min(this.gridWidth - 1, array_x));
        array_y = Math.max(0, Math.min(this.gridHeight - 1, array_y));
  
        // Pass array coordinates directly to setHelitackPoint
        const quenchedCells = this.setHelitackPoint(array_x, array_y);
        const cells2D = this.generateFireStatusMapFromCells(this.engine?.cells ?? [], this.gridWidth, this.gridHeight);
  
        const cellsBurning = this.engine?.cells.filter((cell) => cell.fireState === FireState.Burning).length;
        let done = false;
        if ((!this.simulationRunning && this.engine?.fireDidStop) || cellsBurning === 0) {
          done = true;
        }
        let on_fire = this.isHelicopterOnFire(cells2D, array_x, array_y);
        const response = {
          cells: cells2D,
          done,
          cellsBurning,
          cellsBurnt: this.engine?.cells.filter((cell) => cell.fireState === FireState.Burnt).length,
          quenchedCells,
          on_fire
        };
  
        try {
          if (this.socket?.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(response));
            console.log("✅ Step response sent (action 4)");
          }
        } catch (e) {
          console.log(e);
        }
      }
  }
  @action
  private handleOtherActions(helicopter_coord: number[]) {
      // Use coordinates directly - Python already sends them as [x, y]
      let array_x = helicopter_coord[0];  // First element is x
      let array_y = helicopter_coord[1];  // Second element is y
  
      // Ensure coordinates are within bounds
      array_x = Math.max(0, Math.min(this.gridWidth - 1, array_x));
      array_y = Math.max(0, Math.min(this.gridHeight - 1, array_y));
  
      const cells2D = this.generateFireStatusMapFromCells(this.engine?.cells ?? [], this.gridWidth, this.gridHeight);
  
      const cellsBurning = this.engine?.cells.filter((cell) => cell.fireState === FireState.Burning).length;
      const done = !this.simulationRunning && this.engine?.fireDidStop;
      let on_fire = this.isHelicopterOnFire(cells2D, array_x, array_y);
      const response = {
        cells: cells2D,
        done,
        cellsBurning,
        cellsBurnt: this.engine?.cells.filter((cell) => cell.fireState === FireState.Burnt).length,
        quenchedCells: 0,
        on_fire
      };
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify(response));
        console.log("✅ Step response sent");
      }
  }

  // Cleanup method to close the socket when no longer needed
  @action
  public cleanup() {
    this.shouldReconnect = false;
    if (this.socket) {
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
  }

  @computed public get ready() {
    return this.dataReady && this.sparks.length > 0;
  }

  @computed public get gridWidth() {
    return this.config.gridWidth;
  }

  @computed public get gridHeight() {
    return this.config.gridHeight;
  }

  @computed public get simulationAreaAcres() {
    // dimensions in feet, convert sqft to acres
    return this.config.modelWidth * this.config.modelHeight / 43560;
  }

  @computed public get timeInHours() {
    return Math.floor(this.time / 60);
  }

  @computed public get timeInDays() {
    return this.time / 1440;
  }

  @computed public get canAddSpark() {
    return this.remainingSparks > 0;
  }

  @computed public get zonesCount(): 2 | 3 {
    return this.zones.length as 2 | 3;
  }

  @computed public get remainingSparks() {
    // There's an assumption that number of sparks should be smaller than number of zones.
    return this.zonesCount - this.sparks.length;
  }

  @computed public get canAddFireLineMarker() {
    // Only one fire line can be added at given time.
    if (!this.config.fireLineAvailable) {
      return false;
    }
    else {
      return this.fireLineMarkers.length < 2 && this.time - this.lastFireLineTimestamp > this.config.fireLineDelay;
    }
  }

  @computed public get canUseHelitack() {
    if (!this.config.helitackAvailable) {
      return false;
    }
    else {
      // Helitack has waiting period before it can be used subsequent times
      return this.time - this.lastHelitackTimestamp > this.config.helitackDelay;
    }
  }

  public getZoneBurnPercentage(zoneIdx: number) {
    const burnedCells = this.engine?.burnedCellsInZone[zoneIdx] || 0;
    return burnedCells / this.totalCellCountByZone[zoneIdx];
  }

  public cellAt(x: number, y: number) {
    const gridX = Math.floor(x / this.config.cellSize);
    const gridY = Math.floor(y / this.config.cellSize);
    return this.cells[getGridIndexForLocation(gridX, gridY, this.config.gridWidth)];
  }

  @action.bound public setInputParamsFromConfig() {
    const config = this.config;
    console.log(config);
    
    console.log(config.zones);
    
    this.zones = config.zones.map(options => new Zone(options));
  //   this.zones = Array.from({ length: 18 }, (_, i) => new Zone({
  //   terrainType: i,
  //   name: `Class ${i}`,
  //   vegetationDensity: 1,
  //   terrainRoughness: 1,
  //   fuelLoad: 1,
  // }));

    
    if (config.zonesCount) {
      this.zones.length = config.zonesCount;
    }
    this.zoneIndex = config.zoneIndex || DEFAULT_ZONE_DIVISION[this.zones.length as (2 | 3)];

    this.wind = {
      speed: config.windSpeed,
      direction: config.windDirection
    };
    this.sparks.length = 0;
    config.sparks.forEach(s => {
      this.addSpark(s[0], s[1]);
    });
  }

  @action.bound private configure(presetConfig: Partial<ISimulationConfig>) {
    this.restart();
    // Configuration are joined together. Default values can be replaced by preset, and preset values can be replaced
    // by URL parameters.
    this.config = Object.assign(getDefaultConfig(), presetConfig, getUrlConfig());
    this.setInputParamsFromConfig();
  }

  @action.bound public load(presetConfig: Partial<ISimulationConfig>) {
    this.configure(presetConfig);
    this.populateCellsData();
  }

  @action.bound public populateCellsData() {
    this.dataReady = false;
    const config = this.config;
    const zones = this.zones;
    console.log(zones);
    this.totalCellCountByZone = {};
    this.dataReadyPromise = Promise.all([
      getLandCoverZoneIndex(config), getElevationData(config, zones), getRiverData(config)
    ]).then(values => {
      if (this.backendDriven) {
        this.dataReady = true;
        return;
      }
      const zoneIndex = values[0];
      console.log(zoneIndex);
      const elevation = values[1];
      const river = null; // Removing the river for now
      // const unburntIsland = values[3];

      this.cells.length = 0;

      for (let y = 0; y < this.gridHeight; y++) {
        for (let x = 0; x < this.gridWidth; x++) {
          const index = getGridIndexForLocation(x, y, this.gridWidth);
          const zi = zoneIndex ? zoneIndex[index] : 0;
          // console.log(zi);
          // console.log(zones[zi]);
          
          // const isRiver = river && river[index] > 0;
          // When fillTerrainEdge is set to true, edges are set to elevation 0.
          const isEdge = config.fillTerrainEdges &&
            (x === 0 || x === this.gridWidth - 1 || y === 0 || y === this.gridHeight);
          // Also, edges and their neighboring cells need to be marked as nonburnable to avoid fire spreading over
          // the terrain edge. Note that in this case two cells need to be marked as nonburnable due to way how
          // rendering code is calculating colors for mesh faces.
          const isNonBurnable = config.fillTerrainEdges &&
            x <= 1 || x >= this.gridWidth - 2 || y <= 1 || y >= this.gridHeight - 2;
          const cellOptions: CellOptions = {
            x, y,
            zone: zones[zi],
            zoneIdx: zi,
            baseElevation: isEdge ? 0 : elevation?.[index]
          };
          if (!this.totalCellCountByZone[zi]) {
            this.totalCellCountByZone[zi] = 1;
          } else {
            this.totalCellCountByZone[zi]++;
          }
          this.cells.push(new Cell(cellOptions));
        }
      }
      this.updateCellsElevationFlag();
      this.updateCellsStateFlag();
      this.dataReady = true;
    });
    
  }

  @action.bound public start() {
    if (!this.ready) {
      return;
    }
    if (!this.simulationStarted) {
      this.simulationStarted = true;
    }
    if (!this.engine) {
      this.engine = new FireEngine(this.cells, this.wind, this.sparks, this.config);
      console.log(this.engine);
      
    }

    this.applyFireLineMarkers();

    this.simulationRunning = true;
    this.prevTickTime = null;

    requestAnimationFrame(this.rafCallback);
    console.log(this.cells.length)
  }

  @action.bound public stop() {
    this.simulationRunning = false;
  }

  @action.bound public restart() {
    this.simulationRunning = false;
    this.simulationStarted = false;
    this.cells.forEach(cell => cell.reset());
    this.fireLineMarkers.length = 0;
    this.lastFireLineTimestamp = -Infinity;
    this.lastHelitackTimestamp = -Infinity;
    this.updateCellsStateFlag();
    this.updateCellsElevationFlag();
    this.time = 0;
    this.engine = null;
    this.windDidChange = false;
    if (this.userDefinedWind) {
      this.wind.speed = this.userDefinedWind.speed;
      this.wind.direction = this.userDefinedWind.direction;
      // Clear the saved wind settings. Otherwise, the following scenario might fail:
      // - simulation is started, userDefinedWind is saved when the wind settings are updated during the simulation
      // - user restarts simulation, userDefinedWind props are restored (as expected)
      // - user manually updates wind properties to new values
      // - simulation started and then restarted again BEFORE the new wind settings are applied
      // If userDefinedWind value isn't cleared, the user would see wrong wind setting after the second model restart.
      // This use case is coved by one of the tests in the simulation.test.ts
      this.userDefinedWind = undefined;
    }
  }

  @action.bound public reload() {
    this.restart();
    // Reset user-controlled properties too.
    this.setInputParamsFromConfig();
    // this.load(presetConfig)
    // debugger;
    this.populateCellsData();
    // debugger;
  }

  @action.bound public rafCallback(time: number) {
    
    if (!this.simulationRunning || !this.gymAllowedContinue) {
      return;
    }
    requestAnimationFrame(this.rafCallback);

    let realTimeDiffInMinutes = null;
    if (!this.prevTickTime) {
      this.prevTickTime = time;
    } else {
      
      realTimeDiffInMinutes = (time - this.prevTickTime) / 60000;
      this.prevTickTime = time;
    }
    let timeStep;
    if (realTimeDiffInMinutes) {
      // One day in model time (86400 seconds) should last X seconds in real time.
      const ratio = 86400 / this.config.modelDayInSeconds;
      // Optimal time step assumes we have stable 60 FPS:
      // realTime = 1000ms / 60 = 16.666ms
      // timeStepInMs = ratio * realTime
      // timeStepInMinutes = timeStepInMs / 1000 / 60
      // Below, these calculations are just simplified (1000 / 60 / 1000 / 60 = 0.000277):
      const optimalTimeStep = ratio * 0.000277;
      // Final time step should be limited by:
      // - maxTimeStep that model can handle
      // - reasonable multiplication of the "optimal time step" so user doesn't see significant jumps in the simulation
      //   when one tick takes much longer time (e.g. when cell properties are recalculated after adding fire line)
      timeStep = Math.min(this.config.maxTimeStep, optimalTimeStep * 4, ratio * realTimeDiffInMinutes);
    } else {
      // We don't know performance yet, so simply increase time by some safe value and wait for the next tick.
      timeStep = 1;
    }

    this.tick(timeStep);

    
    // Usage:
    // const gridWidth = 240;
    // const gridHeight = 160;
    // const cells2D = this.reshapeTo2D(this.engine?.cells ?? [], this.gridWidth, this.gridHeight);
    // console.log(cells2D);
    
    // this.socket.send(JSON.stringify({
    //   "ignitionTimes":JSON.stringify(cells2D),
    //   "print":this.time
    // }))
    this.gymAllowedContinue = false
    // this.simulationRunning = false
  }

  @action.bound private reshapeTo2D(cells: any[], width: number, height: number): any[][] {
    const grid: any[][] = [];
  
    for (let row = 0; row < height; row++) {
      const start = row * width;
      const end = start + width;
      const rowIgnitionTimes = cells.slice(start, end).map(cell => {
        if(cell.fireState === FireState.Burnt){
          return -1
        }else if(cell.ignitionTime !== Infinity){
          return cell.ignitionTime
        }else{
          return 0
        }
      });
      grid.push(rowIgnitionTimes);
    }
  
    return grid;
  }

  /**
 * Generates a 2D fire status map from a flat list of cells.
 * Each cell encodes both fire state and burn index as:
 * fire_status = fireState * 3 + burnIndex
 */
@action.bound
private generateFireStatusMapFromCells(cells: any[], width: number, height: number): number[][] {
  const fireStatusMap: number[][] = [];

  for (let row = 0; row < height; row++) {
    const start = row * width;
    const end = start + width;

    const rowData = cells.slice(start, end).map(cell => {
      const fireState = cell.fireState ?? FireState.Unburnt;
      const burnIndex = cell.burnIndex ?? BurnIndex.Low;
      return fireState * 3 + burnIndex;
    });

    fireStatusMap.push(rowData);
  }
  // console.log(fireStatusMap);
  
  return fireStatusMap;
}

@action.bound
private isHelicopterOnFire(fireStatusMap: number[][], array_x: number, array_y: number): boolean {
  // Add bounds checking
  if (!fireStatusMap || array_y < 0 || array_y >= fireStatusMap.length || 
      array_x < 0 || array_x >= fireStatusMap[0].length) {
    console.warn(`Invalid coordinates: x=${array_x}, y=${array_y}, map size=${fireStatusMap?.length}x${fireStatusMap[0]?.length}`);
    return false;
  }
  
  const fireStatus = fireStatusMap[array_y][array_x];
  const fireState = Math.floor(fireStatus / 3);
  return fireState === FireState.Burning;
}

  @action.bound public tick(timeStep: number) {

    if (this.engine) {
      this.time += timeStep;
      this.engine.updateFire(this.time);
      if (this.engine.fireDidStop) {
        this.simulationRunning = false;
        console.log(timeStep);
        console.log(this.time);
      }
    }

    this.updateCellsStateFlag();

    this.changeWindIfNecessary();
    // console.log(this.time,this.engine);
    // let burntCells = this.engine?.cells.filter(cell => cell.fireState === FireState.Burnt)
    // let burningCells = this.engine?.cells
    // const index = getGridIndexForLocation(204, 78, this.gridWidth)
    // console.log(this.time,this.cells[index]);
    
    
  }

  @action.bound public changeWindIfNecessary() {
    if (this.config.changeWindOnDay !== undefined && this.timeInDays >= this.config.changeWindOnDay && this.windDidChange === false) {
      const newDirection = this.config.newWindDirection !== undefined ? this.config.newWindDirection : Math.random() * 360;
      const newSpeed = (this.config.newWindSpeed !== undefined ? this.config.newWindSpeed : Math.random() * NEW_WIND_MAX_SPEED) * this.config.windScaleFactor;
      // Save user defined values that will be restored when model is reset or reloaded.
      this.userDefinedWind = {
        speed: this.wind.speed,
        direction: this.wind.direction
      };
      // Update UI.
      this.wind.direction = newDirection;
      this.wind.speed = newSpeed;
      // Update engine.
      if (this.engine) {
        this.engine.wind.direction = newDirection;
        this.engine.wind.speed = newSpeed;
      }
      // Mark that the change just happened.
      this.windDidChange = true;
    }
  }

  @action.bound public updateCellsElevationFlag() {
    this.cellsElevationFlag += 1;
  }

  @action.bound public updateCellsStateFlag() {
    this.cellsStateFlag += 1;
  }

  @action.bound public addSpark(x: number, y: number) {
    if (this.canAddSpark) {
      this.sparks.push(new Vector2(x, y));
    }
  }

  // Coords are in model units (feet).
  @action.bound public setSpark(idx: number, x: number, y: number) {
    this.sparks[idx] = new Vector2(x, y);
  }

  @action.bound public addFireLineMarker(x: number, y: number) {
    if (this.canAddFireLineMarker) {
      this.fireLineMarkers.push(new Vector2(x, y));
      const count = this.fireLineMarkers.length;
      if (count % 2 === 0) {
        this.markFireLineUnderConstruction(this.fireLineMarkers[count - 2], this.fireLineMarkers[count - 1], true);
      }
    }
  }

  @action.bound public setFireLineMarker(idx: number, x: number, y: number) {
    if (idx % 2 === 1 && idx - 1 >= 0) {
      // Erase old line.
      this.markFireLineUnderConstruction(this.fireLineMarkers[idx - 1], this.fireLineMarkers[idx], false);
      // Update point.
      this.fireLineMarkers[idx] = new Vector2(x, y);
      this.limitFireLineLength(this.fireLineMarkers[idx - 1], this.fireLineMarkers[idx]);
      // Draw a new line.
      this.markFireLineUnderConstruction(this.fireLineMarkers[idx - 1], this.fireLineMarkers[idx], true);
    }
    if (idx % 2 === 0 && idx + 1 < this.fireLineMarkers.length) {
      this.markFireLineUnderConstruction(this.fireLineMarkers[idx], this.fireLineMarkers[idx + 1], false);
      this.fireLineMarkers[idx] = new Vector2(x, y);
      this.limitFireLineLength(this.fireLineMarkers[idx + 1], this.fireLineMarkers[idx]);
      this.markFireLineUnderConstruction(this.fireLineMarkers[idx], this.fireLineMarkers[idx + 1], true);
    }
  }

  @action.bound public markFireLineUnderConstruction(start: ICoords, end: ICoords, value: boolean) {
    const startGridX = Math.floor(start.x / this.config.cellSize);
    const startGridY = Math.floor(start.y / this.config.cellSize);
    const endGridX = Math.floor(end.x / this.config.cellSize);
    const endGridY = Math.floor(end.y / this.config.cellSize);
    forEachPointBetween(startGridX, startGridY, endGridX, endGridY, (x: number, y: number, idx: number) => {
      if (idx % 2 === 0) {
        // idx % 2 === 0 to make dashed line.
        this.cells[getGridIndexForLocation(x, y, this.gridWidth)].isFireLineUnderConstruction = value;
      }
    });
    this.updateCellsStateFlag();
  }

  // Note that this function modifies "end" point coordinates.
  @action.bound public limitFireLineLength(start: ICoords, end: ICoords) {
    const dRatio = dist(start.x, start.y, end.x, end.y) / this.config.maxFireLineLength;
    if (dRatio > 1) {
      end.x = start.x + (end.x - start.x) / dRatio;
      end.y = start.y + (end.y - start.y) / dRatio;
    }
  }

  @action.bound public applyFireLineMarkers() {
    if (this.fireLineMarkers.length === 0) {
      return;
    }
    for (let i = 0; i < this.fireLineMarkers.length; i += 2) {
      if (i + 1 < this.fireLineMarkers.length) {
        this.markFireLineUnderConstruction(this.fireLineMarkers[i], this.fireLineMarkers[i + 1], false);
        this.buildFireLine(this.fireLineMarkers[i], this.fireLineMarkers[i + 1]);
      }
    }
    this.fireLineMarkers.length = 0;
    this.updateCellsStateFlag();
    this.updateCellsElevationFlag();
  }

  @action.bound public buildFireLine(start: ICoords, end: ICoords) {
    const startGridX = Math.floor(start.x / this.config.cellSize);
    const startGridY = Math.floor(start.y / this.config.cellSize);
    const endGridX = Math.floor(end.x / this.config.cellSize);
    const endGridY = Math.floor(end.y / this.config.cellSize);
    forEachPointBetween(startGridX, startGridY, endGridX, endGridY, (x: number, y: number) => {
      const cell = this.cells[getGridIndexForLocation(x, y, this.gridWidth)];
      cell.isFireLine = true;
      cell.ignitionTime = Infinity;
    });
    this.lastFireLineTimestamp = this.time;
  }

  @action.bound public setHelitackPoint(array_x: number, array_y: number) {
    console.log(`Helitack coordinates: (${array_x}, ${array_y})`);
    
    // Use array coordinates directly
    const startGridX = array_x;
    const startGridY = array_y;
    
    // Validate coordinates
    if (startGridX < 0 || startGridX >= this.gridWidth || 
        startGridY < 0 || startGridY >= this.gridHeight) {
        console.warn(`Invalid helitack coordinates: (${startGridX}, ${startGridY})`);
        return 0;
    }
    
    const cellIndex = getGridIndexForLocation(startGridX, startGridY, this.gridWidth);
    const cell = this.cells[cellIndex];
    
    if (!cell) {
        console.warn(`Cell not found at index ${cellIndex} for coordinates (${startGridX}, ${startGridY})`);
        return 0;
    }
    
    const radius = Math.round(this.config.helitackDropRadius / this.config.cellSize);
    let quenchedCells = 0;
    
    for (let x = cell.x - radius; x < cell.x + radius; x++) {
      for (let y = cell.y - radius ; y <= cell.y + radius; y++) {
        if ((x - cell.x) * (x - cell.x) + (y - cell.y) * (y - cell.y) <= radius * radius) {
          const nextCellX = cell.x - (x - cell.x);
          const nextCellY = cell.y - (y - cell.y);
          if (nextCellX >= 0 && nextCellX < this.gridWidth && nextCellY >= 0 && nextCellY < this.gridHeight) {
            const targetCellIndex = getGridIndexForLocation(nextCellX, nextCellY, this.gridWidth);
            const targetCell = this.cells[targetCellIndex];
            if (targetCell) {
              targetCell.helitackDropCount++;
              targetCell.ignitionTime = Infinity;
              if (targetCell.fireState === FireState.Burning) {
                targetCell.fireState = FireState.Unburnt;
                quenchedCells++;
              }
            }
          }
        }
      }
    }
    this.lastHelitackTimestamp = this.time;
    return quenchedCells;
}

  @action.bound public setWindDirection(direction: number) {
    this.wind.direction = direction;
  }

  @action.bound public setWindSpeed(speed: number) {
    this.wind.speed = speed;
  }

  @action.bound public updateZones(zones: Zone[]) {
    this.zones = zones.map(z => z.clone());
    this.zoneIndex = DEFAULT_ZONE_DIVISION[this.zones.length as (2 | 3)];
    if (this.sparks.length > this.zones.length) {
      this.sparks.length = this.zones.length;
    }
    this.populateCellsData();
  }
}
