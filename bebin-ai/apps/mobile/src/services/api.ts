import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = 'http://10.0.2.2:8000'; // Default for Android Emulator
const TOKEN_KEY = 'bebin_auth_token';

const api = axios.create({
  baseURL: API_BASE_URL,
});

api.interceptors.request.use(async (config) => {
  const token = await AsyncStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  use_rag?: boolean;
  use_tools?: boolean;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
}

export const chatApi = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post('/chat', request);
    return response.data;
  },

  // Basic Auth
  login: async (email: string, password: string) => {
    const response = await api.post('/auth/login', { email, password });
    await AsyncStorage.setItem(TOKEN_KEY, response.data.token);
    return response.data;
  }
};

export default api;
