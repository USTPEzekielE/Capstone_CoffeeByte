import React from 'react';

import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import ListScreen from './list';
import ScanScreen from './scan';
import ProfileScreen from './profile';
import { createTabBarIcon } from '../../components';

const Tab = createBottomTabNavigator();

export default function TabLayout() {
  const ListTabBarIcon = createTabBarIcon('list', 'list-outline');
  const ScanTabBarIcon = createTabBarIcon('leaf', 'leaf-outline');
  const HomeTabBarIcon = createTabBarIcon('home', 'home-outline');

  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: 'white',
        tabBarInactiveTintColor: 'rgba(255, 255, 255, 0.6)',
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#3E735B',
          height: 84,
          borderTopLeftRadius: 32,
          borderTopRightRadius: 32,
        },
        tabBarItemStyle: {
          paddingBottom: 12,
          paddingTop: 12,
        },
      }}>
      <Tab.Screen
        name="list"
        component={ListScreen}
        options={{
          title: 'List',
          tabBarIcon: ListTabBarIcon,
        }}
      />
      <Tab.Screen
        name="scan"
        component={ScanScreen}
        options={{
          title: 'Scan',
          tabBarIcon: ScanTabBarIcon,
        }}
      />
      <Tab.Screen
        name="profile"
        component={ProfileScreen}
        options={{
          title: 'Profile',
          tabBarIcon: HomeTabBarIcon,
        }}
      />
    </Tab.Navigator>
  );
}
