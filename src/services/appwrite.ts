import {
  Account,
  Client,
  Databases,
  Functions,
  Storage,
} from 'react-native-appwrite';
const client = new Client();

client
  .setEndpoint(process.env.APPWRITE_CLIENT!)
  .setProject(process.env.APPWRITE_PROJECT_ID!)
  .setPlatform('com.coffeebyte');

const accounts = new Account(client);
const database = new Databases(client);
const storage = new Storage(client);
const functions = new Functions(client);

export { accounts, database, storage, functions };
