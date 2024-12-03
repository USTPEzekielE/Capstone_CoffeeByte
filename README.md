# React Native Vision AI

A React Native application using TensorFlow Lite for computer vision tasks.

## Prerequisites

- Android SDK
  - Build Tools v34.0.0
  - SDK v34
  - NDK v26.1.10
- JDK 21
- Yarn package manager
- TensorFlow dependencies
  - TensorFlow Lite
  - TensorFlow Compiler

## Key Dependencies

- react-native
- react-native-fast-tflite
- react-native-vision-camera

## Installation

1. Clone the repository
2. Copy `env-copy` to `.env` and configure your API keys
3. Install dependencies:
   ```bash
   yarn install
   ```
4. Copy the TensorFlow folder to:
   ```
   node_modules/react-native-fast-tflite/cpp
   ```

## Running the App

1. Connect your Android device via USB or wireless ADB
2. Start the development server:
   ```bash
   yarn start
   ```
3. Press 'a' to run on Android

## Troubleshooting

If you encounter build issues:
- Ensure all prerequisites are installed with correct versions
- Verify Android SDK path in local.properties
- Check ADB connection status