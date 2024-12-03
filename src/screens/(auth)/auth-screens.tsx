import { createNativeStackNavigator } from '@react-navigation/native-stack';
import React from 'react';
import SignIn from './sign-in';

export default function AuthScreens() {
  const Stack = createNativeStackNavigator();

  return (
    <Stack.Navigator
      initialRouteName="sign-in"
      screenOptions={{ headerShown: false }}>
      <Stack.Screen name="sign-in" component={SignIn} />
    </Stack.Navigator>
  );
}
