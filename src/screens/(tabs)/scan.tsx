import { CustomContainer } from '@/components';
import React, { useState, useEffect, useRef, useContext } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { Button, Snackbar } from 'react-native-paper';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { determineResult } from '@/functions/determine-scan';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { EncodingType, readAsStringAsync } from 'expo-file-system';
import { getPrediction } from '@/functions/classifier';
import { LeafList, useLeaf } from '@/providers/leaf-provider';
import { AuthContext } from '@/providers/auth-provider';
import { TypedArray } from '@/types';

interface ModelProps {
  segmentation: {
    resnet: TensorflowModel;
    disease: TensorflowModel;
    biotic: TensorflowModel;
  };
  classification: TensorflowModel;
}

interface ScanResult {
  segmentation: {
    type: 'normal' | 'disease' | 'biotic';
    confidence: number;
    details?: string;
  };
  classifier: {
    class: string;
    confidence: number | bigint;
    index: number;
  };
}

export default function ScanScreen() {
  const [isModelLoading, setIsModelLoading] = useState(true);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [models, setModels] = useState<ModelProps>();
  const [results, setResults] = useState<ScanResult | null>(null);
  const [image, setImage] = useState<string | undefined>();
  const [resizedImage, setReizedImage] = useState('');
  const [snackbarVisible, setSnackBarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const context = useLeaf();
  const user = useContext(AuthContext);

  // Load models on component mount
  useEffect(() => {
    async function loadModels() {
      setIsModelLoading(true);
      try {
        const resnetModel = await loadTensorflowModel(
          require('../../../assets/models/segmentation/resnet.tflite'),
        );
        const diseaseModel = await loadTensorflowModel(
          require('../../../assets/models/segmentation/disease.tflite'),
        );
        const bioticModel = await loadTensorflowModel(
          require('../../../assets/models/segmentation/biotic.tflite'),
        );

        const adagradModel = await loadTensorflowModel(
          require('../../../assets/models/classification/adagrad.tflite'),
        );

        setModels({
          classification: adagradModel,
          segmentation: {
            resnet: resnetModel,
            disease: diseaseModel,
            biotic: bioticModel,
          },
        });
        setIsModelLoading(false);
      } catch (error) {
        console.error('Error loading models:', error);
      }
    }

    loadModels();
  }, []);

  const base64ToUint8Array = (base64: string): Uint8Array => {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  };

  const resizeImage = async (imagePath: string) => {
    try {
      const manipulateResult = await manipulateAsync(
        imagePath,
        [
          {
            resize: {
              width: 256,
              height: 512,
            },
          },
        ],
        {
          compress: 0.8,
          format: SaveFormat.JPEG,
        },
      );
      // Convert the resized image to base64
      const base64 = await readAsStringAsync(manipulateResult.uri, {
        encoding: EncodingType.Base64,
      });

      // Convert base64 to Uint8Array
      const uint8Array = base64ToUint8Array(base64);

      return {
        ...manipulateResult,
        base64,
        uint8Array,
      };
    } catch (error) {
      console.error('Error resizing image:', error);
      throw error;
    }
  };

  const createImageFromTypedArray = (
    typedArray: TypedArray,
    width: number,
    height: number,
  ) => {
    // Convert the TypedArray to Uint8Array if it isn't already
    const uint8Array = new Uint8Array(typedArray.length);

    // If the array contains floats (0-1), convert to 0-255 range
    if (
      typedArray instanceof Float32Array ||
      typedArray instanceof Float64Array
    ) {
      for (let i = 0; i < typedArray.length; i++) {
        uint8Array[i] = Math.floor(typedArray[i] * 255);
      }
    } else {
      uint8Array.set(typedArray);
    }

    // Create RGBA data (4 bytes per pixel)
    const rgbaArray = new Uint8Array(width * height * 4);
    for (let i = 0; i < uint8Array.length; i++) {
      const value = uint8Array[i];
      // Set RGB to the same value for grayscale, and alpha to 255 (fully opaque)
      rgbaArray[i * 4] = value; // R
      rgbaArray[i * 4 + 1] = value; // G
      rgbaArray[i * 4 + 2] = value; // B
      rgbaArray[i * 4 + 3] = 255; // A
    }

    // Convert to base64
    let binary = '';
    for (let i = 0; i < rgbaArray.length; i++) {
      binary += String.fromCharCode(rgbaArray[i]);
    }
    const base64 = btoa(binary);

    return `data:image/png;base64,${base64}`;
  };

  const handleScan = async () => {
    if (!camera.current || !models!.segmentation.resnet || isProcessing) {
      return;
    }

    try {
      setIsProcessing(true);

      // Capture photo
      const photo = await camera.current.takePhoto({
        flash: 'off',
        enableAutoRedEyeReduction: false,
      });
      const resizedPhoto = await resizeImage(`file://${photo.path}`);

      setImage(`file://${photo.path}`);

      if (image === undefined) {
        setReizedImage(resizedPhoto.base64);

        const resnetResults = await models!.segmentation.resnet.run([
          resizedPhoto.uint8Array,
        ]);
        const diseaseResults = await models!.segmentation.disease.run([
          resizedPhoto.uint8Array,
        ]);
        const bioticResults = await models!.segmentation.biotic.run([
          resizedPhoto.uint8Array,
        ]);
        const adagradResults = await models!.classification.run([
          resizedPhoto.uint8Array,
        ]);

        console.log('Processing Classifier');
        const classifier = getPrediction(adagradResults);

        if (
          resnetResults !== null &&
          diseaseResults !== null &&
          bioticResults !== null
        ) {
          const finalResult = determineResult(
            resnetResults,
            diseaseResults,
            bioticResults,
          );

          setResults({ segmentation: finalResult, classifier: classifier });
        }
      } else {
        setSnackBarVisible(true);
        setSnackbarMessage('Not an image leaf, please try again.');
        clean();
      }
    } catch (error) {
      console.error('Error during scan:', error);
      throw error;
    } finally {
      setIsProcessing(false);
    }
  };

  const getSeverity = (percentage: number) => {
    if (percentage < 10) {
      return 'Low';
    }
    if (percentage < 25) {
      return 'Mild';
    }
    if (percentage < 50) {
      return 'Moderate';
    }
    return 'Severe';
  };

  const clean = () => {
    setResults(null);
    setImage(undefined);
  };

  if (!hasPermission) {
    requestPermission();
  }

  if (isModelLoading) {
    return (
      <CustomContainer>
        <View className="flex-1 items-center justify-center">
          <Text>Loading...</Text>
        </View>
      </CustomContainer>
    );
  }

  const onDismissSnackBar = () => setSnackBarVisible(false);

  return (
    <View className="flex-1 flex-col items-center justify-center rounded-3xl my-4">
      <View className="w-[400px] h-[450px] mt-16 items-center">
        {image === undefined ? (
          <Camera
            ref={camera}
            photo={true}
            style={StyleSheet.absoluteFill}
            device={device!}
            isActive={true}
            resizeMode="contain"
          />
        ) : (
          <Image
            source={{ uri: image }}
            style={{ width: 350, height: 350, borderRadius: 50 }}
          />
        )}
      </View>

      {results !== null && (
        <View className="flex flex-col gap-4">
          <Text className="font-medium text-primary text-3xl">
            Disease Name:{' '}
            <Text className="font-normal">{results?.classifier.class}</Text>
          </Text>
          <Text className="font-medium text-primary text-3xl">
            Severity Name:{' '}
            <Text className="font-normal">
              {((results?.segmentation.confidence as number) * 100).toFixed(2) +
                '%'}
            </Text>
          </Text>
          <Text className="font-medium text-primary text-3xl">
            Rating Label:{' '}
            <Text className="font-normal">
              {getSeverity((results?.segmentation.confidence as number) * 100)}
            </Text>
          </Text>
        </View>
      )}

      {results !== null && (
        <View className="flex flex-row items-center justify-center gap-4 w-full mt-6">
          <Button
            mode="contained"
            onPress={clean}
            disabled={isProcessing}
            buttonColor="#3E735B"
            textColor="#fff">
            Re-scan
          </Button>

          <Button
            mode="contained"
            onPress={async () => {
              const leaf: LeafList = {
                image: resizedImage,
                diseasename: results!.classifier.class,
                severity: (results?.segmentation.confidence as number) * 100,
                label: getSeverity(
                  (results?.segmentation.confidence as number) * 100,
                ),
                user: user.user!.id,
              };

              await context.addLeaf(leaf);
              clean();
            }}
            disabled={isProcessing}
            buttonColor="#3E735B"
            textColor="#fff">
            Save
          </Button>
        </View>
      )}

      {results === null && (
        <Text className="mt-16">
          {isProcessing ? 'Processing...' : 'Ready to scan'}
        </Text>
      )}

      {image === undefined && (
        <View className="w-full px-16 mt-32 mb-32">
          <Button
            mode="contained"
            onPress={handleScan}
            disabled={isProcessing}
            buttonColor="#3E735B"
            textColor="#fff">
            {isProcessing ? 'Processing...' : 'Scan'}
          </Button>
        </View>
      )}

      <Snackbar
        visible={snackbarVisible}
        onDismiss={onDismissSnackBar}
        className="text-slate-800 dark:text-slate-100">
        {snackbarMessage}
      </Snackbar>
    </View>
  );
}
