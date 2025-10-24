import axios from "axios";
import type { GameStatus, GameServerStats, ServerMetrics, ReqKillScreenResponse } from "@/types";

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
  // Get device mapping config
  getDeviceMapping: async (): Promise<any> => {
    const response = await apiClient.get("/device-mapping");
    return response.data;
  },

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

  // Test functions for development
  testGameState: async (deviceId: string): Promise<GameStatus> => {
    const response = await apiClient.get<GameStatus>("/gamestate", {
      headers: { "x-device-id": deviceId }
    });
    return response.data;
  },

  testUploadImage: async (deviceId: string, imageId: string, imageBase64: string): Promise<any> => {
    const response = await apiClient.post("/images/upload", {
      image_id: imageId,
      image_data: imageBase64,
      event_type: "ImageUpload"
    }, {
      headers: { "x-device-id": deviceId }
    });
    return response.data;
  },

  testTagPlayer: async (deviceId: string, tagIds: string[], imageIds: string[]): Promise<any> => {
    const response = await apiClient.post("/events", {
      tag_ids: tagIds,
      image_ids: imageIds,
      event_type: "PlayerTagged"
    }, {
      headers: { "x-device-id": deviceId }
    });
    return response.data;
  },


  testKillShotEvent: async (deviceId: string, _displayName: string, _imageBase64: string): Promise<ReqKillScreenResponse> => {
    // displayName and imageBase64 are not used - server retrieves stored images from previous tagging events
    const response = await apiClient.post<ReqKillScreenResponse>("/events", {
      event_type: "ReqKillScreen"
    }, {
      headers: { "x-device-id": deviceId }
    });
    return response.data;
  },

  // Reset all game state and images
  resetGame: async (): Promise<any> => {
    const response = await apiClient.post("/reset");
    return response.data;
  },

  // Test UDP broadcast (simulates device tagging via UDP)
  testUDPBroadcast: async (tagIds: string[], imageIds: string[]): Promise<any> => {
    const response = await apiClient.post("/test/udp-broadcast", {
      tag_ids: tagIds,
      image_ids: imageIds
    });
    return response.data;
  },

  // Test tag player with BOTH HTTP and UDP (simulates real Python behavior)
  testTagPlayerWithUDP: async (deviceId: string, tagIds: string[], imageIds: string[]): Promise<any> => {
    // Send both HTTP event AND UDP broadcast (same as comms_http.py send_tagging_event)
    const [httpResult, udpResult] = await Promise.all([
      apiClient.post("/events", {
        tag_ids: tagIds,
        image_ids: imageIds,
        event_type: "PlayerTagged"
      }, {
        headers: { "x-device-id": deviceId }
      }),
      apiClient.post("/test/udp-broadcast", {
        tag_ids: tagIds,
        image_ids: imageIds
      })
    ]);
    
    return {
      http: httpResult.data,
      udp: udpResult.data
    };
  },
};
