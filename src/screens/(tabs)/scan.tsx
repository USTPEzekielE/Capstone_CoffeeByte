import React, { useState, useRef, useContext } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';
import { Button, Snackbar } from 'react-native-paper';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { LeafList, useLeaf } from '@/providers/leaf-provider';
import { AuthContext } from '@/providers/auth-provider';
import { Result } from '@/types/result';
import { processResults } from '@/functions/process-result';
import { resizeImage } from '@/functions/resize-image';
import { analyzeImage } from '@/functions/analyze-image';

interface ProcessResult {
  disease: {
    class: string;
    confidence: number;
  };
  severities: {
    className: string;
    confidence: number;
  };
}

export default function ScanScreen() {
  const context = useLeaf();
  const user = useContext(AuthContext);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [image, setImage] = useState<string | undefined>();
  const [result, setResult] = useState<Result>();
  const [processResult, setProcessResult] = useState<ProcessResult>();

  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const onDismissSnackBar = () => setSnackbarVisible(false);

  const handleScan = async () => {
    if (!camera.current || isProcessing) {
      return;
    }

    try {
      const photo = await camera.current.takePhoto({
        flash: 'off',
        enableAutoRedEyeReduction: false,
      });

      setImage(photo.path);

      const resizedPhoto = await resizeImage(`file://${photo.path}`);

      const response = await analyzeImage(resizedPhoto.base64);

      if (response !== undefined) {
        setResult(response);

        const process = processResults(response.results);
        setProcessResult(process);
      }
    } catch (error) {
      throw error;
    }
  };

  const clean = () => {
    setResult(undefined);
    setImage(undefined);
    setProcessResult(undefined);
  };

  if (!hasPermission) {
    requestPermission();
  }

  return (
    <View className="flex-1 flex-col items-center justify-center rounded-3xl my-4">
      {image === undefined && (
        <View className="w-[400px] h-[450px] mt-16 items-center">
          <Camera
            ref={camera}
            photo={true}
            style={StyleSheet.absoluteFill}
            device={device!}
            isActive={true}
            resizeMode="contain"
          />
        </View>
      )}

      {image !== undefined && result === undefined && (
        <Image
          source={{ uri: `file://${image}` }}
          className="w-[400px] h-[450px] mt-16 items-center rounded-md p-4"
        />
      )}

      {result !== undefined && (
        <Image
          source={{ uri: result?.image }}
          className="w-[400px] h-[450px] mt-16 items-center rounded-md p-4 mb-16"
        />
      )}

      {processResult !== undefined && (
        <View className="flex flex-col gap-4">
          <Text className="font-medium text-primary text-3xl">
            Disease Name:{' '}
            <Text className="font-normal">{processResult.disease.class}</Text>
          </Text>
          <Text className="font-medium text-primary text-3xl">
            Severity:{' '}
            <Text className="font-normal">
              {((processResult.severities.confidence as number) * 100).toFixed(
                2,
              ) + '%'}
            </Text>
          </Text>
          <Text className="font-medium text-primary text-3xl">
            Rating Label:{' '}
            <Text className="font-normal">
              {processResult.severities.className}
            </Text>
          </Text>
        </View>
      )}

      {result !== undefined && (
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
              if (processResult !== undefined) {
                try {
                  const leaf: LeafList = {
                    image: result.image,
                    diseasename: processResult.disease.class,
                    severity: processResult.severities.confidence * 100,
                    label: processResult.severities.className,
                    user: user.user!.id,
                  };

                  await context.addLeaf(leaf);
                  clean();
                } catch (error) {
                  setSnackbarVisible(true);
                  setSnackbarMessage(`Error saving result: ${error}`);
                }
              }
            }}
            disabled={isProcessing}
            buttonColor="#3E735B"
            textColor="#fff">
            Save
          </Button>
        </View>
      )}

      {result === undefined && (
        <Text className="mt-16">
          {isProcessing ? 'Processing...' : 'Ready to scan'}
        </Text>
      )}

      {result === undefined && (
        <View className="w-full px-16 mt-32 mb-32">
          <Button
            mode="contained"
            onPress={async () => {
              setIsProcessing(true);
              try {
                await handleScan();
              } catch (error) {
                setSnackbarVisible(true);
                setSnackbarMessage(`Error during scan: ${error}`);
                clean();
              }
              setIsProcessing(false);
            }}
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
