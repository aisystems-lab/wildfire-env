import { RefObject, useRef, useState } from "react";
import * as THREE from "three";
import { useThree } from "@react-three/fiber";
import { useStores } from "../../use-stores";

interface UseDraggingInput {
  // If true, this will prevent object from initial jumping between its previous
  // position and point where user grabbed the object.
  useOffset: boolean;
  dragPlane: RefObject<THREE.Mesh | undefined | null> | undefined | null,
  onDrag?: (point: THREE.Vector3) => void
  onDragEnd?: () => void
}

// This helper can be used by interactions.
export const useDragging = ({ useOffset, dragPlane, onDrag, onDragEnd }: UseDraggingInput) => {
  const { raycaster, camera, mouse } = useThree();
  const { ui } = useStores();
  const [dragged, setDragged] = useState(false);
  const offset = useRef<THREE.Vector2 | null>(null);

  const pointerMoveHandler = useRef(() => {
    if (!dragPlane?.current || !onDrag) {
      return;
    }
    if (offset.current) {
      const mouseWithOffset = new THREE.Vector2(mouse.x + offset.current.x, mouse.y + offset.current.y);
      raycaster.setFromCamera(mouseWithOffset, camera);
    }
    const result = raycaster.intersectObject(dragPlane.current);
    if (result?.[0]) {
      onDrag(result[0].point);
    }
  });

  const pointerUpHandler = useRef(() => {
    window.removeEventListener("pointermove", pointerMoveHandler.current);
    window.removeEventListener("pointerup", pointerUpHandler.current);
    ui.dragging = false; // necessary to re-enable orbit controls
    setDragged(false);
    if (onDragEnd) {
      onDragEnd();
    }
  });

  return {
    dragged,
    startDragging: (e: THREE.Event) => {
      if (useOffset) {
        // Calculate difference between actual object position and mouse position. Quite often user won't
        // grab object exactly at its anchor point. This will prevent object from initial jumping between its previous
        // position and point where user grabbed the object.
        const projectedPosition = e.object.position.clone().project(camera);
        offset.current = new THREE.Vector2(projectedPosition.x - mouse.x, projectedPosition.y - mouse.y);
      }
      e.stopPropagation();
      if (!dragPlane?.current) {
        return;
      }
      window.addEventListener("pointermove", pointerMoveHandler.current);
      window.addEventListener("pointerup", pointerUpHandler.current);
      ui.dragging = true; // necessary to disable orbit controls
      setDragged(true);
    }
  };
};
