import React, { forwardRef, useLayoutEffect, useRef } from "react";
import { BurnIndex, Cell, FireState } from "../../models/cell";
import { ISimulationConfig } from "../../config";
import * as THREE from "three";
import { BufferAttribute } from "three";
import { SimulationModel } from "../../models/simulation";
import { ftToViewUnit, PLANE_WIDTH, planeHeight } from "./helpers";
import { observer } from "mobx-react";
import { useStores } from "../../use-stores";
import { getEventHandlers, InteractionHandler } from "./interaction-handler";
import { usePlaceSparkInteraction } from "./use-place-spark-interaction";
import { useShowCoordsInteraction } from "./use-show-coords-interaction";
import { useHelitackInteraction } from "./use-helitack-interaction";

const vertexIdx = (cell: Cell, gridWidth: number, gridHeight: number) => (gridHeight - 1 - cell.y) * gridWidth + cell.x;

// const getTerrainColor = (droughtLevel: number) => {
//   switch (droughtLevel) {
//     case DroughtLevel.NoDrought:
//       return [0.008, 0.831, 0.039];
//     case DroughtLevel.MildDrought:
//       return [0.573, 0.839, 0.216];
//     case DroughtLevel.MediumDrought:
//       return [0.757, 0.886, 0.271];
//     default:
//       return [0.784, 0.631, 0.271];
//   }
// };

const getTerrainColor = (landcoverType: number): [number, number, number] => {
  switch (landcoverType) {
    case 1: return [0.0, 0.392, 0.0];       // Evergreen Needleleaf Forest
    case 2: return [0.133, 0.545, 0.133];   // Evergreen Broadleaf Forest
    case 3: return [0.196, 0.804, 0.196];   // Deciduous Needleleaf Forest
    case 4: return [0.0, 0.5, 0.0];         // Deciduous Broadleaf Forest
    case 5: return [0.235, 0.702, 0.443];   // Mixed Forest
    case 6: return [0.941, 0.902, 0.549];   // Closed Shrublands
    case 7: return [0.855, 0.647, 0.125];   // Open Shrublands
    case 8: return [0.502, 0.502, 0.0];     // Woody Savannas
    case 9: return [0.604, 0.804, 0.196];   // Savannas
    case 10: return [0.565, 0.933, 0.565];  // Grasslands
    case 11: return [0.561, 0.737, 0.561];  // Permanent Wetlands
    case 12: return [0.824, 0.706, 0.549];  // Croplands
    case 13: return [0.502, 0.502, 0.502];  // Urban and Built-Up
    case 14: return [0.871, 0.722, 0.529];  // Cropland/Natural Vegetation Mosaic
    case 15: return [1.0, 1.0, 1.0];        // Snow and Ice
    case 16: return [0.827, 0.827, 0.827];  // Barren or Sparsely Vegetated
    case 17: return [0.0, 0.0, 1.0];        // Water
    default: return [0.5, 0.5, 0.5];        // Default: Grey
  }
};
const BURNING_COLOR = [1, 0, 0];
const BURNT_COLOR = [0.2, 0.2, 0.2];
const FIRE_LINE_UNDER_CONSTRUCTION_COLOR = [0.5, 0.5, 0];
const HELITACK_COLOR = [0.6, 0, 0.6]; // purple
const HELICOPTER_DEBUG_CELL_COLOR = [0.0, 1.0, 1.0];

export const BURN_INDEX_LOW = [1, 0.7, 0];
export const BURN_INDEX_MEDIUM = [1, 0.5, 0];
export const BURN_INDEX_HIGH = [1, 0, 0];

const burnIndexColor = (burnIndex: BurnIndex) => {
  if (burnIndex === BurnIndex.Low) {
    return BURN_INDEX_LOW;
  }
  if (burnIndex === BurnIndex.Medium) {
    return BURN_INDEX_MEDIUM;
  }
  return BURN_INDEX_HIGH;
};

const setVertexColor = (
  colArray: number[],
  cell: Cell,
  gridWidth: number,
  gridHeight: number,
  config: ISimulationConfig,
  helicopterGridCoord?: [number, number]
) => {
  const idx = vertexIdx(cell, gridWidth, gridHeight) * 4;
  let color;
  if (helicopterGridCoord && cell.x === helicopterGridCoord[0] && cell.y === helicopterGridCoord[1]) {
    color = HELICOPTER_DEBUG_CELL_COLOR;
  } else if (cell.helitackDropCount > 0) {
    color = HELITACK_COLOR;
  } else if (cell.fireState === FireState.Burning) {
    color = config.showBurnIndex ? burnIndexColor(cell.burnIndex) : BURNING_COLOR;
  } else if (cell.fireState === FireState.Burnt) {
    color = cell.isFireSurvivor ? getTerrainColor(cell.vegetation) : BURNT_COLOR;
  } else if (cell.isRiver) {
    color = config.riverColor;
  } else if (cell.isFireLineUnderConstruction) {
    color = FIRE_LINE_UNDER_CONSTRUCTION_COLOR;
  } else {
    color = getTerrainColor(cell.vegetation);
  }
  // Note that we're using sRGB colorspace here (default while working with web). THREE.js needs to operate in linear
  // color space, so we need to convert it first. See:
  // https://discourse.threejs.org/t/updates-to-color-management-in-three-js-r152/50791
  // https://threejs.org/docs/#manual/en/introduction/Color-management
  const threeJsColor = new THREE.Color();
  threeJsColor.setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);

  colArray[idx] = threeJsColor.r;
  colArray[idx + 1] = threeJsColor.g;
  colArray[idx + 2] = threeJsColor.b;
  colArray[idx + 3] = 1; // alpha
};

const updateColors = (geometry: THREE.PlaneGeometry, simulation: SimulationModel) => {
  const colArray = geometry.attributes.color.array as number[];
  simulation.cells.forEach(cell => {
    setVertexColor(
      colArray,
      cell,
      simulation.gridWidth,
      simulation.gridHeight,
      simulation.config,
      simulation.helicopterGridCoord
    );
  });
  (geometry.attributes.color as BufferAttribute).needsUpdate = true;
};

const setupElevation = (geometry: THREE.PlaneGeometry, simulation: SimulationModel) => {
  const posArray = geometry.attributes.position.array as number[];
  const mult = ftToViewUnit(simulation);
  // Apply height map to vertices of plane.
  simulation.cells.forEach(cell => {
    const zAttrIdx = vertexIdx(cell, simulation.gridWidth, simulation.gridHeight) * 3 + 2;
    posArray[zAttrIdx] = cell.elevation * mult;
  });
  geometry.computeVertexNormals();
  (geometry.attributes.position as BufferAttribute).needsUpdate = true;
};

export const Terrain = observer(forwardRef<THREE.Mesh>(function WrappedComponent(props, ref) {
  const { simulation } = useStores();
  const height = planeHeight(simulation);
  const geometryRef = useRef<THREE.PlaneGeometry>(null);

  useLayoutEffect(() => {
    geometryRef.current?.setAttribute("color",
      new THREE.Float32BufferAttribute(new Array((simulation.gridWidth) * (simulation.gridHeight) * 4), 4)
    );
  }, [simulation.gridWidth, simulation.gridHeight]);


  useLayoutEffect(() => {
    if (geometryRef.current) {
      setupElevation(geometryRef.current, simulation);
    }
  }, [simulation, simulation.cellsElevationFlag]);

  useLayoutEffect(() => {
    if (geometryRef.current) {
      updateColors(geometryRef.current, simulation);
    }
  }, [simulation, simulation.cellsStateFlag]);

  const interactions: InteractionHandler[] = [
    usePlaceSparkInteraction(),
    useShowCoordsInteraction(),
    useHelitackInteraction()
  ];

  // Note that getEventHandlers won't return event handlers if it's not necessary. This is important,
  // as adding even an empty event handler enables raycasting machinery in @react-three/fiber and it has big
  // performance cost in case of fairly complex terrain mesh. That's why when all the interactions are disabled,
  // eventHandlers will be an empty object and nothing will be attached to the terrain mesh.
  const eventHandlers = getEventHandlers(interactions);

  return (
    /* eslint-disable react/no-unknown-property */
    // See: https://github.com/jsx-eslint/eslint-plugin-react/issues/3423
    <mesh
      ref={ref}
      position={[PLANE_WIDTH * 0.5, height * 0.5, 0]}
      {...eventHandlers}
    >
      <planeGeometry
        attach="geometry"
        ref={geometryRef}
        center-x={0} center-y={0}
        args={[PLANE_WIDTH, height, simulation.gridWidth - 1, simulation.gridHeight - 1]}
      />
      <meshPhongMaterial attach="material" vertexColors={true} />
    </mesh>
    /* eslint-enable react/no-unknown-property */
  );
}));
