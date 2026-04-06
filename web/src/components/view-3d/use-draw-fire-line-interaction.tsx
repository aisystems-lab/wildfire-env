import { useRef } from "react";
import { useStores } from "../../use-stores";
import { ftToViewUnit } from "./helpers";
import { Event } from "three";
import { Interaction } from "../../models/ui";
import { useDragging } from "./use-dragging";
import * as THREE from "three";
import { InteractionHandler } from "./interaction-handler";

const MIN_DIST = 1500; // feet

export const useDrawFireLineInteraction: () => InteractionHandler = () => {
  const { simulation, ui } = useStores();
  const dragPlane = useRef<THREE.Mesh>();

  const { startDragging } = useDragging({
    useOffset: false,
    dragPlane,
    onDrag: (point: THREE.Vector3) => {
      const ratio = ftToViewUnit(simulation);
      const x = point.x / ratio;
      const y = point.y / ratio;
      const lastIdx = simulation.fireLineMarkers.length - 1;
      simulation.setFireLineMarker(lastIdx, x, y);
    },
    onDragEnd: () => {
      const lastIdx = simulation.fireLineMarkers.length - 1;
      if (
        Math.abs(simulation.fireLineMarkers[lastIdx - 1].x - simulation.fireLineMarkers[lastIdx].x) < MIN_DIST &&
        Math.abs(simulation.fireLineMarkers[lastIdx - 1].y - simulation.fireLineMarkers[lastIdx].y) < MIN_DIST
      ) {
        // Markers are too close, it was a click probably. Reset them and keep interaction on.
        simulation.markFireLineUnderConstruction(
          simulation.fireLineMarkers[lastIdx - 1], simulation.fireLineMarkers[lastIdx], false
        );
        simulation.fireLineMarkers.length = simulation.fireLineMarkers.length - 2;
      } else {
        // Markers are fine, finish interaction.
        ui.interaction = null;
      }
    }
  });

  return {
    active: ui.interaction === Interaction.DrawFireLine,
    onPointerDown: (e: Event) => {
      const ratio = ftToViewUnit(simulation);
      const x = e.point.x;
      const y = e.point.y;
      simulation.addFireLineMarker(x / ratio, y / ratio);
      simulation.addFireLineMarker(x / ratio, y / ratio);
      // There's assumption that user will click on terrain mesh to start drawing fire line.
      dragPlane.current = e.object as THREE.Mesh;
      startDragging(e);
    }
  };
};
