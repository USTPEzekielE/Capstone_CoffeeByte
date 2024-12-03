import React from 'react';
import CustomText from './custom-text';
import CustomTextInput from './inputs/text-input';
import { View } from 'react-native';

interface Props {
  label: string;
  value: string;
}

export default function ProfileInfoContainer({ label, value }: Props) {
  return (
    <View className="font-serif w-full">
      <CustomText>{label}</CustomText>
      <CustomTextInput
        placeholder={value}
        placeholderTextColor="#3E735B"
        className="mt-2 rounded-lg border-[2px] border-[#3E735B]"
      />
    </View>
  );
}
