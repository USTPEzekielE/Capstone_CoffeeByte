import { ReactNode } from 'react';
import { User } from './user_type';

interface ProviderProps {
  children: ReactNode;
}

interface AuthContextProps {
  user: User | null;
  isAuthenticated: boolean;
  addUser: (user: User) => void;
  login: (email: string, password: string) => Promise<any>;
  logout: () => void;
}

export type { ProviderProps, AuthContextProps };
