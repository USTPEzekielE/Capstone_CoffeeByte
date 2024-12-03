import { TypedArray } from '@/types';
import RNFS from 'react-native-fs';

// Helper function to convert TypedArrays to string representation
const convertTypedArrayToString = (arr: TypedArray | null): string => {
  if (!arr) {
    return '';
  }
  // Convert the typed array to regular array and join with semicolons
  return Array.from(arr).join(';');
};

// Helper function to convert array of Float32Arrays to string
// const convertFloat32ArraysToString = (arrays: Float32Array[]): string => {
//   return arrays.map(arr => Array.from(arr).join(';')).join('|');
// };

// Helper function to convert Uint8Array to base64 string with chunking
const convertUint8ArrayToString = (arr: Uint8Array): string => {
  // Simply join the values with semicolons
  return Array.from(arr).join(';');
};

const convertUint8ArrayToBase64 = (arr: Uint8Array): string => {
  try {
    // Convert to regular array
    const bytes = Array.from(arr);
    let binary = '';
    const chunkSize = 1024; // Process 1KB at a time

    // Process the array in chunks to avoid stack overflow
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.slice(i, i + chunkSize);
      binary += String.fromCharCode(...chunk);
    }

    // Convert binary string to base64
    return btoa(binary);
  } catch (error) {
    console.error('Error converting to base64:', error);

    // Alternative approach if the above fails
    try {
      // Convert numbers to strings directly
      return arr.join(',');
    } catch (fallbackError) {
      console.error('Fallback conversion failed:', fallbackError);
      return '';
    }
  }
};

const saveResultsToCSV = async (
  resnetResults: TypedArray[] | null,
  diseaseResults: TypedArray[] | null,
  bioticResults: TypedArray[] | null,
  adagradResults: TypedArray[] | null,
  inputArray: string,
  imageData: Uint8Array,
  fileName = 'scan_results.csv',
) => {
  try {
    const path = `${RNFS.DocumentDirectoryPath}/${fileName}`;

    const resnetString = resnetResults
      ? resnetResults.map(convertTypedArrayToString).join('|')
      : '';
    const diseaseString = diseaseResults
      ? diseaseResults.map(convertTypedArrayToString).join('|')
      : '';
    const bioticString = bioticResults
      ? bioticResults.map(convertTypedArrayToString).join('|')
      : '';
    const adagradString = adagradResults
      ? adagradResults.map(convertTypedArrayToString).join('|')
      : '';
    const inputArrayString = inputArray;

    // Handle image data conversion with progress logging
    console.log('Starting image data conversion...');
    const imageDataString = convertUint8ArrayToString(imageData);
    console.log('Image data conversion completed');

    // Create CSV row
    const row = [
      resnetString,
      'ResNet',
      diseaseString,
      'Disease',
      bioticString,
      'Biotic',
      adagradString,
      'Adagrad',
      inputArrayString,
      'Float32Array[]',
      imageDataString,
      'Uint8Array',
    ].join(',');

    // Combine headers and data
    const csvContent = row;

    // Write to file
    await RNFS.writeFile(path, csvContent, 'utf8');

    console.log(`Results saved to: ${path}`);
    return path;
  } catch (error) {
    console.error('Error saving results to CSV:', error);
    throw error;
  }
};

const appendResultsToCSV = async (
  resnetResults: TypedArray[] | null,
  diseaseResults: TypedArray[] | null,
  bioticResults: TypedArray[] | null,
  adagradResults: TypedArray[] | null,
  inputData: string,
  imageData: Uint8Array,
  fileName = 'scan_results.csv',
) => {
  try {
    const path = `${RNFS.DocumentDirectoryPath}/${fileName}`;
    const fileExists = await RNFS.exists(path);

    const timestamp = new Date().toISOString();

    // Convert arrays to strings
    const resnetString = resnetResults
      ? resnetResults.map(convertTypedArrayToString).join('|')
      : '';
    const diseaseString = diseaseResults
      ? diseaseResults.map(convertTypedArrayToString).join('|')
      : '';
    const bioticString = bioticResults
      ? bioticResults.map(convertTypedArrayToString).join('|')
      : '';
    const adagradString = adagradResults
      ? adagradResults.map(convertTypedArrayToString).join('|')
      : '';
    const base64 = inputData;
    const uint8list = convertUint8ArrayToString(imageData);
    const uint8Base64 = convertUint8ArrayToBase64(imageData);

    // Create new row
    const newRow = [
      timestamp,
      resnetString,
      'ResNet',
      diseaseString,
      'Disease',
      bioticString,
      'Biotic',
      adagradString,
      'Adagrad',
      base64,
      'Base64',
      uint8list,
      'Uint8Array',
      'ToBase64',
      uint8Base64,
    ].join(',');

    if (!fileExists) {
      await RNFS.writeFile(path, newRow, 'utf8');
    } else {
      // Append new row to existing file
      await RNFS.appendFile(path, `\n${newRow}`, 'utf8');
    }

    console.log(`Results appended to: ${path}`);
    return path;
  } catch (error) {
    console.error('Error appending results to CSV:', error);
    throw error;
  }
};

export { saveResultsToCSV, appendResultsToCSV };
