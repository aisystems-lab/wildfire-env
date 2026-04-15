/**
 * Accepts height and width of the expected grid array, as well as the image with any dimensions.
 * Returns 1D grid array populated with data from image 2D array.
 *
 * Note that there's an assumption that image origin (0, 0) is in its top-left corner, while grid
 * has its origin (0, 0) in bottom-left corner.
 *
 * When interpolate parameter is true, it will use linear interpolation to get missing values.
 */
export const populateGrid = (width: number, height: number, image: number[][], interpolate = false): number[] => {
  const arr = [];
  // Figure out the size of the image using the first row.
  const imageHeight = image.length;
  const imageWidth = image[0].length;
  const numGridCellsPerImageRowPixel = interpolate ? (imageHeight - 1) / (height - 1) : imageHeight / height;
  const numGridCellsPerImageColPixel = interpolate ? (imageWidth - 1) / (width - 1) : imageWidth / width;

  let imageRowIndex = imageHeight - 1;
  let imageRowAdvance = 0;
  for (let r = 0; r < height; r++) {
    let imageColIndex = 0;
    let imageColAdvance = 0;
    for (let c = 0; c < width; c++) {
      let value = image[imageRowIndex][imageColIndex];
      if (interpolate) {
        // Bi-linear interpolation.
        const bottomLeft = image[imageRowIndex][imageColIndex];
        const bottomRight = imageColIndex + 1 < imageWidth ? image[imageRowIndex][imageColIndex + 1] : bottomLeft;
        const topLeft = imageRowIndex - 1 >= 0 ? image[imageRowIndex - 1][imageColIndex] : bottomLeft;
        const topRight = imageRowIndex - 1 >= 0 ?
          (imageColIndex + 1 < imageWidth ? image[imageRowIndex - 1][imageColIndex + 1] : topLeft) : bottomRight;
        value = bottomLeft * (1 - imageColAdvance) * (1 - imageRowAdvance) +
                bottomRight * imageColAdvance * (1 - imageRowAdvance) +
                topLeft * (1 - imageColAdvance) * imageRowAdvance +
                topRight * imageColAdvance * imageRowAdvance;
      }
      arr.push(value);
      imageColAdvance += numGridCellsPerImageColPixel;
      if (imageColAdvance >= 1) {
        imageColIndex += Math.floor(imageColAdvance);
        imageColAdvance -= Math.floor(imageColAdvance);
      }
    }
    imageRowAdvance += numGridCellsPerImageRowPixel;
    if (imageRowAdvance >= 1) {
      imageRowIndex -= Math.floor(imageRowAdvance);
      imageRowAdvance -= Math.floor(imageRowAdvance);
    }
  }
  return arr;
};

// Returns transformed image data.
export const getImageData = (
  imgSrc: string,
  // Function that transfers [r, g, b, a] array into single value
  mapColor: (rgba: [number, number, number, number]) => number,
  // Final callback when data is ready and processed.
  callback: (imgData: number[][]) => void) => {
  interface IImageLoadedEvent {
    target: EventTarget | null;
  }

  const imageLoaded = (event: IImageLoadedEvent) => {
    const img = event.target as HTMLImageElement;
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (ctx === null) {
      throw new Error("can't get 2d canvas context");
    }
    ctx.drawImage(img, 0, 0, img.width, img.height);
    const rawData: ImageData = ctx.getImageData(0, 0, img.width, img.height);
    const data: number[][] = [];

    for (let y = 0; y < rawData.height; y += 1) {
      const row: number[] = [];
      data.push(row);
      for (let x = 0; x < rawData.width * 4; x += 4) {
        const rIdx = y * (rawData.width * 4) + x;
        row.push(mapColor([
          rawData.data[rIdx],
          rawData.data[rIdx + 1],
          rawData.data[rIdx + 2],
          rawData.data[rIdx + 3]
        ]));
      }
    }
    callback(data);
  };

  // Load image first.
  const imgage = document.createElement("img");
  imgage.src = imgSrc;
  if (imgage.complete) {
    imageLoaded({target: imgage});
  } else {
    imgage.addEventListener("load", imageLoaded);
    imgage.addEventListener("error", () => {
      throw new Error(`Cannot load image ${imgSrc}`);
    });
  }
};

export const getInputData = (
  input: number[][] | string | undefined,
  gridWidth: number,
  gridHeight: number,
  interpolate: boolean,
  // Function that transfers [r, g, b, a] array into single value
  mapColor: (rgba: [number, number, number, number]) => number,
): Promise<number[] | undefined> => {
  return new Promise(resolve => {
    if (input === undefined) {
      resolve(undefined);
    } else if (input.constructor === Array) {
      resolve(populateGrid(gridWidth, gridHeight, input as number[][], interpolate));
    } else { // input is a string, URL to an image
      getImageData(
        input as string,
        mapColor,
        (imageData: number[][]) => {
          resolve(populateGrid(gridWidth, gridHeight, imageData, interpolate));
        }
      );
    }
  });
};

export const hexToRGBValue = (hex: string, alpha: number) => {
  const convertedHexValue = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  const rgb = convertedHexValue ? {
    r: parseInt(convertedHexValue[1], 16),
    g: parseInt(convertedHexValue[2], 16),
    b: parseInt(convertedHexValue[3], 16)
  } : null;

  if (rgb) {
    return `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
  } else {
    return "rgba(170, 170, 170,1.0)";
  }
};

export const hexToRGB = (hex: string) => {
  const convertedHexValue = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return convertedHexValue ? {
    r: parseInt(convertedHexValue[1], 16),
    g: parseInt(convertedHexValue[2], 16),
    b: parseInt(convertedHexValue[3], 16)
  } : null;
};
