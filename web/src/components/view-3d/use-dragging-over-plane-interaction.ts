import { RefObject, useState } from "react";
import * as THREE from "three";
import { useStores } from "../../use-stores";
import { useDragging } from "./use-dragging";
import { ftToViewUnit } from "./helpers";
import { Event } from "three";
import { Interaction } from "../../models/ui";

export const useDraggingOverPlaneInteraction = (
  enabled: boolean,
  onDrag?: (x: number, y: number) => void,
  onDragEnd?: () => void,
  dragPlane?: RefObject<THREE.Mesh>
) => {
  const { simulation, ui } = useStores();
  const [ hovered, setHover ] = useState(false);
  const { dragged, startDragging } = useDragging({
    useOffset: true,
    dragPlane,
    onDrag: (p: THREE.Vector3) => {
      if (onDrag) {
        const r = ftToViewUnit(simulation);
        onDrag(p.x / r, p.y / r);
      }
    },
    onDragEnd: () => {
      setHover(false);
      onDragEnd?.();
    }
  });
  // This interaction should be active only when all the others are inactive.
  // HoverOverDraggable is set by this handler below.
  const active = enabled && (ui.interaction === null || ui.interaction === Interaction.HoverOverDraggable);
  return {
    active,
    hovered,
    dragged,
    onPointerOver: (e: Event) => {
      // Ignore this event while object is being dragged around. We don't want to trigger some state changes
      // when cursor randomly enters or leaves object area.
      if (!dragged) {
        e.stopPropagation();
        setHover(true);
        ui.interaction = Interaction.HoverOverDraggable;
      }
    },
    onPointerOut: (e: Event) => {
      // Ignore this event while object is being dragged around. We don't want to trigger some state changes
      // when cursor randomly enters or leaves object area.
      if (!dragged) {
        e.stopPropagation();
        setHover(false);
        ui.interaction = null;
      }
    },
    onPointerDown: (e: Event) => {
      e.stopPropagation();
      startDragging(e);
    }
  };
};
