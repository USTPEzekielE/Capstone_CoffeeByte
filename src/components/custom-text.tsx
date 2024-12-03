import React from 'react';
import { Text } from 'react-native';

import clsx from 'clsx';

export default function CustomText(props: React.ComponentProps<typeof Text>) {
  return (
    <Text
      {...props}
      className={clsx(
        !props.className?.includes('text-') &&
          'dark:text-slate-100 text-slate-800',
        props.className,
      )}>
      {props.children}
    </Text>
  );
}
