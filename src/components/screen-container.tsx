import React, { ReactNode } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface CustomContainerProps {
  children: ReactNode;
  scrollEnabled?: boolean;
  enableKeyboardAvoidingView?: boolean;
}

export default function CustomContainer({
  children,
  scrollEnabled = true,
  enableKeyboardAvoidingView = true,
}: CustomContainerProps) {
  const { top, bottom, right, left } = useSafeAreaInsets();
  return (
    <View
      className={`flex-1 pt-[${top}] pb-[${bottom}] pr-[${right}] pl-[${left}]`}>
      <KeyboardAvoidingView
        enabled={enableKeyboardAvoidingView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        className="flex-1">
        <ScrollView
          scrollEnabled={scrollEnabled}
          contentInsetAdjustmentBehavior="automatic"
          keyboardShouldPersistTaps="handled"
          alwaysBounceVertical
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{ flexGrow: 1 }}>
          {children}
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}
