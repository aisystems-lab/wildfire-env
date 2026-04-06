import React from "react";
import { observer } from "mobx-react";
import { useStores } from "../../use-stores";
import { Marker } from "./marker";

const createLabelTexture = (label: string) => {
  const canvas = document.createElement("canvas");
  canvas.width = 192;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return canvas;
  }

  ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111";
  ctx.font = "bold 24px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, canvas.width / 2, canvas.height / 2);
  return canvas;
};

export const CornerLabels: React.FC = observer(function WrappedComponent() {
  const { simulation } = useStores();
  const cellSize = simulation.config.cellSize;
  const maxX = simulation.gridWidth - 1;
  const maxY = simulation.gridHeight - 1;

  const corners = [
    { key: "tl", label: "[0, 0]", x: 0.5 * cellSize, y: (maxY + 0.5) * cellSize },
    { key: "tr", label: `[${maxX}, 0]`, x: (maxX + 0.5) * cellSize, y: (maxY + 0.5) * cellSize },
    { key: "bl", label: `[0, ${maxY}]`, x: 0.5 * cellSize, y: 0.5 * cellSize },
    { key: "br", label: `[${maxX}, ${maxY}]`, x: (maxX + 0.5) * cellSize, y: 0.5 * cellSize },
  ];

  return (
    <>
      {corners.map(corner => (
        <Marker
          key={corner.key}
          markerImg={createLabelTexture(corner.label)}
          position={{ x: corner.x, y: corner.y }}
          width={0.12}
          height={0.04}
          anchorX={0.5}
          anchorY={0.5}
        />
      ))}
    </>
  );
});
