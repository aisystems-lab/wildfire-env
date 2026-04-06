import React, { useMemo, RefObject } from "react";
import * as THREE from "three";
import { observer } from "mobx-react";
import { ftToViewUnit, PLANE_WIDTH } from "./helpers";
import { useStores } from "../../use-stores";
import { getEventHandlers } from "./interaction-handler";
import { useDraggingOverPlaneInteraction } from "./use-dragging-over-plane-interaction";

const getTexture = (imgSrcOrCanvas: string | HTMLCanvasElement) => {
  let source;
  let Texture = THREE.Texture;
  if (typeof imgSrcOrCanvas === "string") {
    source = document.createElement("img");
    source.src = imgSrcOrCanvas;
    source.onload = () => texture.needsUpdate = true;
  } else {
    source = imgSrcOrCanvas; // canvas
    Texture = THREE.CanvasTexture;
  }
  const texture = new Texture(source);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
};

interface IProps {
  // Image src or HTML Canvas that is going to be used as a texture source.
  markerImg: string | HTMLCanvasElement;
  position: {x: number, y: number};
  // Width relative to the plane/terrain width.
  width?: number;
  // Height relative to the plane/terrain width.
  height?: number;
  anchorX?: number;
  anchorY?: number;
  // onDrag and dragPlane enable dragging.
  onDrag?: (x: number, y: number) => void;
  onDragEnd?: () => void;
  dragPlane?: RefObject<THREE.Mesh>;
  // Optional highlight image that we'll be activated on hover.
  markerHighlightImg?: string | HTMLCanvasElement;
  lockOnSimStart?: boolean;
}

export const Marker: React.FC<IProps> = observer(function WrappedComponent({
 markerImg, markerHighlightImg, position, onDrag, onDragEnd, dragPlane,
 width = 0.06, height = 0.06, anchorX = 0.5, anchorY = 0, lockOnSimStart = false
}) {
  const { simulation } = useStores();
  const defTexture = useMemo(() => getTexture(markerImg), [markerImg]);
  const highlightTexture = useMemo(() => markerHighlightImg && getTexture(markerHighlightImg), [markerHighlightImg]);
  const lockedOnSimStart = lockOnSimStart && simulation.simulationStarted;
  // Dragging is disabled when onDrag and dragPlane are missing, or when marker is locked on sim start.
  const draggingEnabled = !!onDrag && !!dragPlane && !lockedOnSimStart;
  const draggingInteraction = useDraggingOverPlaneInteraction(draggingEnabled, onDrag, onDragEnd, dragPlane);

  if (!simulation.dataReady) {
    // Don't render markers when simulation data isn't downloaded yet.
    return null;
  }

  const ratio = ftToViewUnit(simulation);
  const x = position.x * ratio;
  const y = position.y * ratio;
  const z = simulation.cellAt(position.x, position.y).elevation * ratio;

  const scaleMult = lockedOnSimStart ? 0.5 : 1;

  const texture = draggingInteraction.hovered && highlightTexture ? highlightTexture : defTexture;

  const eventHandlers = getEventHandlers([ draggingInteraction ]);
  return (
    /* eslint-disable react/no-unknown-property */
    // See: https://github.com/jsx-eslint/eslint-plugin-react/issues/3423
    <sprite
      renderOrder={1}
      position={[x, y, z]}
      scale={[width * PLANE_WIDTH * scaleMult, height * PLANE_WIDTH * scaleMult, 1]}
      center-x={anchorX}
      center-y={anchorY}
      {...eventHandlers}
    >
      <spriteMaterial attach="material" map={texture} depthTest={false} />
    </sprite>
    /* eslint-enable react/no-unknown-property */
  );
});
