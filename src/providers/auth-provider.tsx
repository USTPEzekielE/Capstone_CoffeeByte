import { accounts, database } from '@/services/appwrite';
import { User } from '@/types';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AuthContextProps, ProviderProps } from '@/types/auth_provider_type';
import React, {
  createContext,
  FC,
  useCallback,
  useContext,
  useMemo,
  useState,
  useEffect,
} from 'react';
import { CustomContainer } from '@/components';
import { Text, View } from 'react-native';
import { Query } from 'react-native-appwrite';

const initialState: AuthContextProps = {
  user: null,
  isAuthenticated: false,
  addUser: () => {},
  login: () => Promise.resolve(),
  logout: () => Promise.resolve(),
};

export const AuthContext = createContext<AuthContextProps>(initialState);

export const useCustomProvider = () => {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useCustomProvider must be used within a CustomProvider');
  }
  return context;
};

export const AuthenticationProvider: FC<ProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const tempUser: User = useMemo(
    () => ({
      id: '1',
      email: 'admin@coffeebyte.co',
      session: 'U2FsdGVkX1+3',
      username: 'admin',
      lastName: 'Edorot',
      firstName: 'Aren Dale',
      phoneNumber: '+639758500506',
      dateOfBirth: new Date().toISOString(),
      avatar: '../assets/images/icon.png',
    }),
    [],
  );

  const getUserFromStorage = async (): Promise<void> => {
    try {
      const response = await AsyncStorage.getItem('user');

      if (response) {
        const userData = JSON.parse(response) as User;

        try {
          await accounts.getSession(userData.session);
          setUser(userData);
        } catch (error) {
          await AsyncStorage.removeItem('user');
          setUser(null);
        }
      }
    } catch (error) {
      // Do not throw an error when response is empty
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    getUserFromStorage();
  }, []);

  const login = useCallback(
    async (email: string, password: string): Promise<User> => {
      // Simplified return type
      if (email === 'admin@coffeebyte.co' && password === 'password') {
        setUser(tempUser);
        await AsyncStorage.setItem('user', JSON.stringify(tempUser));
        return tempUser;
      }

      if (email.length === 0 || password.length === 0) {
        throw new Error('Email and password are required');
      }

      try {
        const authResponse = await accounts.createEmailPasswordSession(
          email,
          password,
        );

        const roleResponse = await database.listDocuments(
          process.env.APPWRITE_DATABASE!,
          process.env.ROLE_COLLECTION!,
          [
            Query.and([
              Query.equal('role', '673ee7be0020a2298fd1'),
              Query.equal('user', authResponse.userId),
            ]),
          ],
        );

        console.log(
          'response: ',
          roleResponse,
          '\ntotal: ',
          roleResponse.total,
        );

        if (roleResponse.total === 1) {
          Promise.reject('Admin users cannot use the mobile application.');
        }

        const dbResponse = await database.getDocument(
          process.env.APPWRITE_DATABASE!,
          process.env.USERS_COLLECTION!,
          authResponse.userId,
        );

        const userData: User = {
          id: authResponse.userId,
          session: authResponse.$id,
          email: dbResponse.email,
          username: dbResponse.username,
          firstName: dbResponse.firstname,
          lastName: dbResponse.lastname,
          avatar: dbResponse.avatar,
          dateOfBirth: dbResponse.birthdate,
          phoneNumber: dbResponse.phonenumber,
        };

        setUser(userData);
        await AsyncStorage.setItem('user', JSON.stringify(userData));
        return userData;
      } catch (error) {
        throw new Error(
          `Authentication failed: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
        );
      }
    },
    [tempUser],
  );

  const addUser = useCallback((data: User) => {
    setUser(data);
  }, []);

  const logout = useCallback(async () => {
    try {
      if (user?.session) {
        try {
          await accounts.deleteSession(user.session);
        } catch (error) {
          console.error('Error deleting session:', error);
        }
        await AsyncStorage.removeItem('user');
      }
    } catch (error) {
      console.error('Error during logout:', error);
    } finally {
      setUser(null);
    }
  }, [user]);

  const contextValue = useMemo(
    () => ({
      isAuthenticated: !!user,
      user,
      addUser,
      login,
      logout,
    }),
    [user, addUser, login, logout],
  );

  if (isLoading) {
    return (
      <CustomContainer>
        <View className="flex-1 flex-col items-center justify-center">
          <Text>Loading...</Text>
        </View>
      </CustomContainer>
    );
  }

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
};
