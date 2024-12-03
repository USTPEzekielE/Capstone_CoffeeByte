module.exports = {
  presets: ['babel-preset-expo', 'nativewind/babel'],
  plugins: [
    [
      'module-resolver',
      {
        root: ['.'],
        extensions: [
          '.ios.ts',
          '.android.ts',
          '.ts',
          '.ios.tsx',
          '.android.tsx',
          '.tsx',
          '.jsx',
          '.js',
          '.json',
          'tflite',
        ],
        alias: {
          '@': './src',
          '@assets': './assets',
          '@components': './src/components',
          '@screens': './src/screens',
          '@functions': './src/functions',
          '@providers': './src/providers',
          '@types/*': 'src/types',
          '@services': 'src/services',
        },
      },
    ],
    [
      'module:react-native-dotenv',
      {
        moduleName: '@env',
        path: '.env',
        safe: true,
        allowUndefined: false,
      },
    ],
    'react-native-reanimated/plugin',
  ],
};
