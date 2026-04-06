import React from "react";
import { observer } from "mobx-react";
import { useStores } from "../../use-stores";
import { Marker } from "./marker";

const font = (size: number) => {
  return `${size}px Lato, arial, helvetica, sans-serif`;
};

const labelCanvas = (label: string) => {
  const width = 512;
  const height = 128;
  const dotRadius = height / 8;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return canvas;
  }
  // Dot
  const lineWidth = width / 64;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  ctx.arc(dotRadius + lineWidth * 0.5, height - dotRadius - lineWidth * 0.5, dotRadius, 0, 2 * Math.PI);
  ctx.strokeStyle = "#797979";
  ctx.fillStyle = "#fff";
  ctx.stroke();
  ctx.fill();
  // Label
  ctx.font = font(height * 0.4);
  ctx.fillText(label, dotRadius * 2.5, height - dotRadius * 2.5, width);
  return canvas;
};

export const TownMarkersContainer: React.FC = observer(function WrappedComponent() {
  const { simulation } = useStores();
  return <>
    {
      simulation.townMarkers.map((town, idx) => {
        return <Marker
          key={idx}
          markerImg={labelCanvas(town.name)}
          position={town.position}
          width={0.2}
          height={0.05}
          anchorX={0.03} // move anchor point to the left, trying to match the dot
        />;
      })
    }
         </>;
});
