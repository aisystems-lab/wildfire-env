import React from "react";
import { observer } from "mobx-react";
import { useStores } from "../../use-stores";
import helitackImg from "../../assets/interactions/helitack-cursor.png";
import { Marker } from "./marker";

export const HelicopterMarker: React.FC = observer(function WrappedComponent() {
  const { simulation } = useStores();

  if (!simulation.backendDriven) {
    return null;
  }

  return (
    <Marker
      markerImg={helitackImg}
      position={simulation.helicopterModelPosition}
      width={0.045}
      height={0.045}
      anchorX={0.5}
      anchorY={0.5}
    />
  );
});
