import MainApp from '@/main-app';
import { AuthenticationProvider } from '@/providers/auth-provider';
import {
  DarkTheme,
  DefaultTheme,
  NavigationContainer,
  ThemeProvider,
} from '@react-navigation/native';
import { useColorScheme } from 'nativewind';
import React from 'react';
import { PaperProvider } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import './global.css';
import { LeafProvider } from '@/providers/leaf-provider';

function App(): React.JSX.Element {
  const colorScheme = useColorScheme();
  const isDark = colorScheme.colorScheme === 'dark';
  return (
    <SafeAreaProvider>
      <PaperProvider>
        <NavigationContainer>
          <ThemeProvider value={isDark ? DarkTheme : DefaultTheme}>
            <AuthenticationProvider>
              <LeafProvider>
                <MainApp />
              </LeafProvider>
            </AuthenticationProvider>
          </ThemeProvider>
        </NavigationContainer>
      </PaperProvider>
    </SafeAreaProvider>
  );
}

export default App;
