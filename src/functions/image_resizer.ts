import { Image } from 'react-native';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

/**
 * Resizes an image from Vision Camera while maintaining aspect ratio
 * @param {string} uri - The URI of the captured image
 * @param {number} targetWidth - Desired width of the resized image
 * @param {number} targetHeight - Desired height of the resized image
 * @param {string} format - Output format ('jpeg' or 'png'), defaults to 'jpeg'
 * @param {number} quality - Image quality (0 to 1), defaults to 0.8
 * @returns {Promise<{ uri: string, width: number, height: number }>}
 */
const resizeVisionCameraImage = async (
  uri: string,
  targetWidth: number,
  targetHeight: number,
  format = 'jpeg',
  quality = 0.8,
) => {
  try {
    // Get the original image dimensions
    const imageInfo: { width: number; height: number } = await new Promise(
      (resolve, reject) => {
        Image.getSize(
          uri,
          (width, height) => resolve({ width, height }),
          error => reject(error),
        );
      },
    );

    // Calculate aspect ratio
    const aspectRatio = imageInfo.width / imageInfo.height;

    // Determine new dimensions while maintaining aspect ratio
    let newWidth = targetWidth;
    let newHeight = targetHeight;

    if (targetWidth / targetHeight > aspectRatio) {
      newWidth = targetHeight * aspectRatio;
    } else {
      newHeight = targetWidth / aspectRatio;
    }

    // Perform the resize operation
    const manipulateResult = await manipulateAsync(
      uri,
      [
        {
          resize: {
            width: Math.round(newWidth),
            height: Math.round(newHeight),
          },
        },
      ],
      {
        compress: quality,
        format: format === 'png' ? SaveFormat.PNG : SaveFormat.JPEG,
      },
    );

    return {
      uri: manipulateResult.uri,
      width: Math.round(newWidth),
      height: Math.round(newHeight),
    };
  } catch (error) {
    console.error('Error resizing image:', error);
    throw error;
  }
};

export default resizeVisionCameraImage;
