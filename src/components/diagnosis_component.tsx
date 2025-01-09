import React from 'react';
import { Image, View } from 'react-native';

import CustomText from './custom-text';
import { LeafList } from '@/providers/leaf-provider';

export default function DiagnosisComponent({
  leaf,
}: {
  leaf: LeafList;
}) {
  const image = leaf.image.includes('data:image/png;base64,') ? leaf.image : `data:image/png;base64,${leaf.image}`;
  return (
    <View className="flex-row items-center border-2 rounded-3xl border-gray-400 py-3 px-4 mb-2">
      <Image
        source={{ uri: image }}
        className="w-12 h-14 rounded-md mr-3"
      />
      <View className="pl-6">
        <CustomText className="font-bold text-primary text-xl">
          {leaf.diseasename}
        </CustomText>
        <CustomText className="text-primary text-base">
          ID: {leaf.id}
        </CustomText>
      </View>
    </View>
  );
}
