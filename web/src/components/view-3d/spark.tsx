import React, { RefObject } from "react";
import { observer } from "mobx-react";
import { useStores } from "../../use-stores";
import sparkImg from "../../assets/interactions/spark.png";
import sparkHighlightImg from "../../assets/interactions/spark-highlight.png";
import { Marker } from "./marker";
import * as THREE from "three";

interface IProps {
  dragPlane: RefObject<THREE.Mesh>
}

export const SparksContainer: React.FC<IProps> = observer(function WrappedComponent({ dragPlane }) {
  const { simulation } = useStores();
  return <>
    {
      simulation.sparks.map((s, idx) => {
        const onDrag = (x: number, y: number) => simulation.setSpark(idx, x, y);
        return <Marker
          key={idx}
          markerImg={sparkImg}
          markerHighlightImg={sparkHighlightImg}
          position={s}
          onDrag={onDrag}
          dragPlane={dragPlane}
          lockOnSimStart={true}
        />;
      })
    }
         </>;
});
