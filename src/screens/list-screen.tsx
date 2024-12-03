import React from 'react';
import { CustomContainer, CustomText, DiagnosisComponent } from '../components';
import { View } from 'react-native';
import { formatDate, groupDiagnosesByDate } from '../functions/helper';
import { useLeaf } from '@/providers/leaf-provider';

export default function ListsScreen() {
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
              <View>
                <DiagnosisComponent leaf={leaf} key={index} />
              </View>
            ))}
          </View>
        ))}
      </View>
    </CustomContainer>
  );
}
