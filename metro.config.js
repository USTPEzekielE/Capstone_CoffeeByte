const { getDefaultConfig } = require('expo/metro-config');
const { mergeConfig } = require('@react-native/metro-config');
const {
  wrapWithReanimatedMetroConfig,
} = require('react-native-reanimated/metro-config');
const { withNativeWind } = require('nativewind/metro');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('metro-config').MetroConfig}
 */
const config = {
  resolver: {
    assetExts: [
      'png',
      'jpg',
      'jpeg',
      'gif',
      'webp',
      'mp4',
      'mp3',
      'wav',
      'ttf',
      'otf',
      'tflite',
    ],
  },
};

module.exports = withNativeWind(
  wrapWithReanimatedMetroConfig(
    mergeConfig(getDefaultConfig(__dirname), config),
  ),
  { input: './global.css' },
);
