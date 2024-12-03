import React, { useMemo, useState } from 'react';
import { View } from 'react-native';
import { Button, Text } from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import CustomText from '../../components/custom-text';
import CustomTextInput from '../../components/inputs/text-input';
import DiagnosisComponent from '../../components/diagnosis_component';
import { formatDate, groupDiagnosesByDate } from '../../functions/helper';
import { useNavigation } from '@react-navigation/native';
import { useLeaf } from '@/providers/leaf-provider';

export default function ListScreen() {
  const navigation = useNavigation();
  const [search, setSearch] = useState('');
  const context = useLeaf();

  // Group diagnoses by date
  const filteredAndGroupedDiagnoses = useMemo(() => {
    const filteredDiagnoses = context.leaves.filter(leaf =>
      leaf.diseasename.toLowerCase().includes(search.toLowerCase()),
    );
    return groupDiagnosesByDate(filteredDiagnoses);
  }, [context.leaves, search]);

  return (
    <SafeAreaView className="flex-1">
      <View className="px-12 flex-1 justify-between">
        <View className="pt-12" />
        <CustomTextInput
          placeholder="Search"
          onChangeText={value => setSearch(value)}
          className="mb-4"
        />

        <CustomText className="text-primary font-bold text-3xl text-center mb-4">
          Recent Diagnosis
        </CustomText>

        <View className="flex-1 overflow-hidden gap-3">
          {Object.entries(filteredAndGroupedDiagnoses).map(([date, leafs]) => (
            <View key={date}>
              <CustomText className="font-bold text-gray-600 mt-2 mb-2">
                {formatDate(date)}
              </CustomText>
              {leafs.map((leaf, index) => (
                <DiagnosisComponent leaf={leaf} key={index} />
              ))}
            </View>
          ))}
        </View>

        <Button
          mode="contained"
          onPress={() => navigation.navigate('lists')}
          buttonColor="#3E735B"
          textColor="#fff"
          className="w-full mt-4">
          See More
        </Button>
      </View>
      <View className="pb-4" />
    </SafeAreaView>
  );
}
