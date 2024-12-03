import React, { useContext } from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import AuthScreens from './screens/(auth)/auth-screens';
import TabLayout from './screens/(tabs)/tabs-layout.tsx';
import { AuthContext } from './providers/auth-provider.tsx';
import ListsScreen from './screens/list-screen.tsx';

export default function MainApp() {
  const Stack = createNativeStackNavigator();
  const user = useContext(AuthContext);
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: false,
      }}>
      {user.user ? (
        <>
          <Stack.Screen name="tabs" component={TabLayout} />
          <Stack.Screen name="lists" component={ListsScreen} />
        </>
      ) : (
        <Stack.Screen name="auth" component={AuthScreens} />
      )}
    </Stack.Navigator>
  );
}
