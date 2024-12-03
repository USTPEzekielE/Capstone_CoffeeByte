import { database } from '@/services/appwrite';
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import { ID, Query } from 'react-native-appwrite';
import { AuthContext } from './auth-provider';

export type LeafList = {
  id?: string;
  image: string;
  diseasename: string;
  severity: number;
  label: string;
  user: string;
  createdAt?: string;
};

interface LeafContextType {
  leaves: LeafList[];
  addLeaf: (leaf: LeafList) => Promise<void>;
  removeLeaf: (index: number) => Promise<void>;
  updateLeaf: (index: number, leaf: LeafList) => Promise<void>;
  clearLeaves: () => Promise<void>;
}

const initialState: LeafContextType = {
  leaves: [],
  addLeaf: () => Promise.resolve(),
  removeLeaf: () => Promise.resolve(),
  updateLeaf: () => Promise.resolve(),
  clearLeaves: () => Promise.resolve(),
};

export const LeafContext = createContext<LeafContextType>(initialState);

export function LeafProvider({ children }: { children: React.ReactNode }) {
  const [leaves, setLeaves] = useState<LeafList[]>([]);
  const user = useContext(AuthContext);

  const getLeafsFromCloud = useCallback(async () => {
    try {
      const response = await database.listDocuments(
        process.env.APPWRITE_DATABASE!,
        process.env.SCAN_IMAGES_COLLECTION!,
        [Query.equal('user', user.user!.id)],
      );

      if (response.total > 0) {
        const leafs = response.documents.map(value => {
          const leaf: LeafList = {
            id: value.$id,
            diseasename: value.diseasename,
            image: value.image,
            label: value.label,
            severity: value.severity,
            user: value.user,
            createdAt: value.$createdAt,
          };
          return leaf;
        });

        setLeaves(leafs);
      }
    } catch (error) {
      console.error('Error retrieving user from storage:', error);
    }
  }, [user.user]);

  useEffect(() => {
    getLeafsFromCloud();
  }, [getLeafsFromCloud]);

  const addLeaf = async (leaf: LeafList): Promise<void> => {
    await database.createDocument(
      process.env.APPWRITE_DATABASE!,
      process.env.SCAN_IMAGES_COLLECTION!,
      ID.unique(),
      leaf,
    );

    const newLeaf: LeafList = {
      ...leaf,
      createdAt: new Date().toISOString(),
    };

    setLeaves(prev => [...prev, newLeaf]);
  };

  const removeLeaf = async (index: number) => {
    setLeaves(prev => prev.filter((_, i) => i !== index));
  };

  const updateLeaf = async (index: number, leaf: LeafList) => {
    setLeaves(prev => prev.map((item, i) => (i === index ? leaf : item)));
  };

  const clearLeaves = async () => {
    setLeaves([]);
  };

  return (
    <LeafContext.Provider
      value={{ leaves, addLeaf, removeLeaf, updateLeaf, clearLeaves }}>
      {children}
    </LeafContext.Provider>
  );
}

export function useLeaf() {
  const context = useContext(LeafContext);
  if (!context) {
    throw new Error('useLeaf must be used within a LeafProvider');
  }
  return context;
}
