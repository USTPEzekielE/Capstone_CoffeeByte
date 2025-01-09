import React, { useContext, useState } from 'react';
import { Image, View } from 'react-native';
import { Button, Snackbar } from 'react-native-paper';
import { AuthContext } from '../../providers/auth-provider';
import CustomContainer from '../../components/screen-container';
import CustomText from '../../components/custom-text';
import CustomTextInput from '../../components/inputs/text-input';
import { AppwriteException } from 'react-native-appwrite';

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const context = useContext(AuthContext);

  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const onDismissSnackBar = () => setSnackbarVisible(false);

  return (
    <CustomContainer>
      <View className="px-6 flex-1 flex-col items-center justify-between">
        <View className="gap-4 flex flex-col items-center justify-center w-full pt-16">
          <Image source={require('../../../assets/images/icon.png')} />

          <CustomText className="text-4xl font-bold my-8 leading-none dark:text-[#81c7a7] text-[#3E735B]">
            Login Now
          </CustomText>

          <CustomTextInput
            autoComplete="email"
            placeholder="Email"
            onChangeText={value => setEmail(value)}
          />

          <CustomTextInput
            autoComplete="password"
            placeholder="Password"
            secureTextEntry={true}
            onChangeText={value => setPassword(value)}
          />
        </View>

        <View className="mt-12 w-full pb-16">
          <Button
            mode="contained"
            onPress={async () => {
              context
                .login(email.toLocaleLowerCase(), password)
                .catch((error: AppwriteException) => {
                  setSnackbarMessage(error.message);
                  setSnackbarVisible(true);
                });
            }}
            buttonColor="#3E735B"
            textColor="#fff">
            Log in
          </Button>
        </View>

        <Snackbar
          visible={snackbarVisible}
          onDismiss={onDismissSnackBar}
          className="text-slate-800 dark:text-slate-100">
          {snackbarMessage}
        </Snackbar>
      </View>
    </CustomContainer>
  );
}
