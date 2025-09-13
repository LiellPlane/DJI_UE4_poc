// Common types used across the application
import { Request } from "express";

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface HealthCheck {
  status: "UP" | "DOWN";
  timestamp: string;
  uptime: number;
  memory: NodeJS.MemoryUsage;
  environment: string;
}

export interface PaginationQuery {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export interface RequestWithUser extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

// Game-specific types matching your Python Pydantic models
// event_type is REQUIRED in all these interfaces (matches Python default_factory)
export interface UploadRequest {
  image_id: string;
  image_data: string; // base64 encoded JPEG
  event_type: string; // REQUIRED - matches Python default_factory
  timestamp?: number;
}

export interface PlayerStatus {
  health: number;
  ammo: number;
  tag_id: string;
  display_name: string;
  event_type: string; // REQUIRED - matches Python default_factory
}

export interface GameUpdate {
  players: Record<string, PlayerStatus>;
  event_type: string; // REQUIRED - matches Python default_factory
}

export interface PlayerTagged {
  tag_id: string;
  image_ids: string[];
  event_type: string; // REQUIRED - matches Python default_factory
}

export interface ImageInfo {
  image_id: string;
  user_id: string;
  timestamp: number;
  size_bytes: number;
  received_at: number;
}

export interface EventInfo {
  event_type: string;
  user_id: string;
  data: any;
  received_at: number;
}

export interface GameServerStats {
  server_info: {
    uptime_seconds: number;
    start_time: string;
  };
  activity: {
    total_images: number;
    total_events: number;
    recent_images: ImageInfo[];
    recent_events: EventInfo[];
  };
  game_state: {
    active_players: number;
    players: Record<string, PlayerStatus>;
  };
}

// Custom request interface with user ID header
export interface GameRequest extends Request {
  headers: Request["headers"] & {
    "x-user-id"?: string;
  };
}
