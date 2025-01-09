import clsx from 'clsx';
import React, { useState } from 'react';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { TextInput, TouchableOpacity, View } from 'react-native';

export default function CustomTextInput(
  props: React.ComponentProps<typeof TextInput>,
) {

  const [secure, setSecure] = useState(props.secureTextEntry);

  return (
    <View className="relative w-full">
      <TextInput
        {...props}
        secureTextEntry={props.secureTextEntry && secure}
        className={clsx(
          `px-4 py-2 text-base w-full 
          ${props.multiline ?? 'h-[58px]'}
          ${!props.className?.includes('border') &&
          'border-2 border-gray-400 rounded-[20px]'
          } 
          ${props.secureTextEntry && 'pr-12'} 
          dark:text-slate-100 text-slate-800`,
          props.className,
        )}
      />
      {props.secureTextEntry && (
        <TouchableOpacity
          className="absolute right-4 h-full justify-center"
          onPress={() => setSecure(!secure)}
        >
          {secure ? (
            <Ionicons name='eye-off-outline' size={20} color="#666" />
          ) : (
            <Ionicons name='eye-outline' size={20} color="#666" />
          )}
        </TouchableOpacity>
      )}
    </View>
  );
}