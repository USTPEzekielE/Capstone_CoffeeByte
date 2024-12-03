import clsx from 'clsx';
import React from 'react';
import { TextInput } from 'react-native';

export default function CustomTextInput(
  props: React.ComponentProps<typeof TextInput>,
) {
  return (
    <TextInput
      {...props}
      className={clsx(
        `px-4 py-2 text-base w-full 
        ${props.multiline ?? 'h-[58px]'}
        ${
          !props.className?.includes('border') &&
          'border-2 border-gray-400 rounded-[20px]'
        } dark:text-slate-100 text-slate-800`,
        props.className,
      )}
    />
  );
}
