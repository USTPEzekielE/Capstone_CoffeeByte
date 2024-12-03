import React from 'react';
import { Image, Text, View } from 'react-native';
import { LeafList } from '@/providers/leaf-provider';

export default function ScanDetailsScreen(leaf: LeafList) {
  return (
    <View className="flex-1 flex-col items-center justify-center rounded-3xl my-4">
      <View className="w-[400px] h-[450px] mt-16 items-center">
        <Image
          source={{ uri: leaf.image }}
          style={{ width: 350, height: 350, borderRadius: 50 }}
        />
      </View>

      <View className="flex flex-col gap-4">
        <Text className="font-medium text-primary text-3xl">
          Disease Name: <Text className="font-normal">{leaf.diseasename}</Text>
        </Text>
        <Text className="font-medium text-primary text-3xl">
          Severity Name:{' '}
          <Text className="font-normal">
            {/* {((results?.segmentation.confidence as number) * 100).toFixed(2) +
              '%'} */}
            {leaf.severity}
          </Text>
        </Text>
        <Text className="font-medium text-primary text-3xl">
          Rating Label: <Text className="font-normal">{leaf.label}</Text>
        </Text>
      </View>
    </View>
  );
}
