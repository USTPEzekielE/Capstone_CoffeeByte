import React from 'react';
import { TouchableOpacity, View } from 'react-native';
import { useLeaf } from '@/providers/leaf-provider';
import { CustomContainer, CustomText, DiagnosisComponent } from '@/components';
import { formatDate, groupDiagnosesByDate } from '@/functions/helper';
import { useNavigation } from '@react-navigation/native';

export default function ListsScreen() {
  const navigation = useNavigation();
  const context = useLeaf();
  const leaves = groupDiagnosesByDate(context.leaves);

  return (
    <CustomContainer>
      <View className="p-6">
        <CustomText className="text-primary font-bold text-3xl text-center">
          Recent Diagnosis
        </CustomText>

        {Object.entries(leaves).map(([date, leafs]) => (
          <View key={date}>
            <CustomText className="font-bold text-gray-600 mt-4 mb-4">
              {formatDate(date)}
            </CustomText>
            {leafs.map((leaf, index) => (
              <TouchableOpacity
              key={`${date}-${index}`}
                onPress={() => navigation.navigate('leaf', { leaf })}>
                <DiagnosisComponent leaf={leaf} key={index} />
              </TouchableOpacity>
            ))}
          </View>
        ))}
      </View>
    </CustomContainer>
  );
}
