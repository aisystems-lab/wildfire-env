// Only four directions. It's important, as it makes less likely the river or fire line is accidentally crossed by the
// fire (e.g. when it's really narrow and drawn at 45* angle).
export const directNeighbours = [ {x: -1, y: 0}, {x: 1, y: 0}, {x: 0, y: -1}, {x: 0, y: 1} ];

export const dist = (x0: number, y0: number, x1: number, y1: number) => {
  return Math.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
};

export const withinDist = (x0: number, y0: number, x1: number, y1: number, maxDist: number) => {
  return (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) <= maxDist * maxDist;
};

export const getGridIndexForLocation = (gridX: number, gridY: number, width: number) => {
  return gridX + gridY * width;
};

// Bresenham's line algorithm.
export const forEachPointBetween = (
  x0: number, y0: number, x1: number, y1: number, callback: (x: number, y: number, idx: number) => void
) => {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = (x0 < x1) ? 1 : -1;
  const sy = (y0 < y1) ? 1 : -1;
  let err = dx - dy;
  let idx = 0;
  for (;;) {
    callback(x0, y0, idx);
    idx += 1;
    if ((x0 === x1) && (y0 === y1)) break;
    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y0 += sy;
    }
  }
};
