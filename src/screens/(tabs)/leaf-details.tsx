import React from 'react';
import { Image, Text, View } from 'react-native';
import { useRoute } from '@react-navigation/native';

export default function LeafDetails() {
  const route = useRoute();
  const { leaf } = route.params;
  const image = leaf.image.includes('data:image/png;base64,')
    ? leaf.image
    : `data:image/png;base64,${leaf.image}`;
  return (
    <View className="flex-1 flex-col items-center justify-center rounded-3xl my-4">
      <Image
        source={{ uri: image }}
        className="w-[400px] h-[450px] mt-16 items-center rounded-md p-4 mb-16"
      />

      <View className="flex flex-col gap-4">
        <Text className="font-medium text-primary text-3xl">
          Disease Name: <Text className="font-normal">{leaf.diseasename}</Text>
        </Text>
        <Text className="font-medium text-primary text-3xl">
          Severity:{' '}
          <Text className="font-normal">{leaf.severity.toFixed(2) + '%'}</Text>
        </Text>
        <Text className="font-medium text-primary text-3xl">
          Rating Label: <Text className="font-normal">{leaf.label}</Text>
        </Text>
      </View>
    </View>
  );
}
