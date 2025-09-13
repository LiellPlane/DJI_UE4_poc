import axios from "axios";
import type { HealthStatus, SystemStatus } from "@/types";

const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string) || "http://localhost:3000/api";

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
    console.log(
      `Making ${config.method?.toUpperCase()} request to ${config.url}`,
    );
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
  // Health check
  getHealth: async (): Promise<HealthStatus> => {
    const response = await apiClient.get<HealthStatus>("/health");
    return response.data;
  },

  // Detailed health check
  getDetailedHealth: async (): Promise<HealthStatus> => {
    const response = await apiClient.get<HealthStatus>("/health?detailed=true");
    return response.data;
  },

  // System status
  getStatus: async (): Promise<SystemStatus> => {
    const response = await apiClient.get<SystemStatus>("/status");
    return response.data;
  },

  // Readiness probe
  getReadiness: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await apiClient.get("/health/ready");
    return response.data;
  },

  // Liveness probe
  getLiveness: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await apiClient.get("/health/live");
    return response.data;
  },
};
