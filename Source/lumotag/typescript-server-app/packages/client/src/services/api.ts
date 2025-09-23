import axios from "axios";
import type { GameStatus, GameServerStats, ServerMetrics } from "@/types";

const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string) || "http://localhost:8080/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Only log non-gamestate requests to reduce noise
    if (!config.url?.includes('/gamestate')) {
      console.log(
        `Making ${config.method?.toUpperCase()} request to ${config.url}`,
      );
    }
    return config;
  },
  (error) => {
    console.error("Request error:", error);
    return Promise.reject(error);
  },
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error("Response error:", error.response?.data || error.message);
    return Promise.reject(error);
  },
);

export const apiService = {
  // Get current game state with all players (dashboard endpoint - no auth required)
  getGameState: async (): Promise<GameStatus> => {
    const response = await apiClient.get<GameStatus>("/dashboard/gamestate");
    return response.data;
  },

  // Get detailed server stats including images and performance
  getStats: async (): Promise<GameServerStats> => {
    const response = await apiClient.get<GameServerStats>("/stats");
    return response.data;
  },

  // Get server metrics (queue status, processing info)
  getMetrics: async (): Promise<ServerMetrics> => {
    const response = await apiClient.get<ServerMetrics>("/metrics");
    return response.data;
  },
};
