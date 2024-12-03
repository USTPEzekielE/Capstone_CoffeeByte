export const prepareInputData = (imageData: Uint8Array) => {
  // Create a Float32Array with the required shape [-1, 256, 512, 3]
  const inputArray = new Float32Array(256 * 512 * 3);

  // Convert and normalize the image data
  for (let i = 0; i < imageData.length; i += 4) {
    // Convert RGBA to RGB and normalize to [-1, 1] or [0, 1] depending on your model
    const r = imageData[i] / 255.0;
    const g = imageData[i + 1] / 255.0;
    const b = imageData[i + 2] / 255.0;

    // Calculate the position in the input array
    const pixelIndex = (i / 4) * 3;
    inputArray[pixelIndex] = r;
    inputArray[pixelIndex + 1] = g;
    inputArray[pixelIndex + 2] = b;
  }

  return [inputArray];
};
