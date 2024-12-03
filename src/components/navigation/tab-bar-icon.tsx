// You can explore the built-in icon families and icons on the web at https://icons.expo.fyi/

import React from 'react';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { IconProps } from 'react-native-vector-icons/Icon';
import { StyleSheet } from 'react-native';

function createTabBarIcon(focusedName: string, unfocusedName: string) {
  return ({ color, focused }: { focused: boolean; color: string }) => {
    return (
      <TabBarIcon name={focused ? focusedName : unfocusedName} color={color} />
    );
  };
}

function TabBarIcon({ style, ...rest }: IconProps) {
  return <Ionicons size={28} style={[styles.icon, style]} {...rest} />;
}

const styles = StyleSheet.create({
  icon: {
    marginBottom: -3,
  },
});

export { createTabBarIcon };
