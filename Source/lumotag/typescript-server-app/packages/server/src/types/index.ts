// Types for the game server
import { Request } from "express";

// Game-specific types matching your Python Pydantic models exactly
// These match the structures in lumotag_events.py
// All fields are REQUIRED - no defaults, crash if missing
export interface UploadRequest {
  image_id: string;
  image_data: string; // base64 encoded JPEG
  event_type: string;
}

export interface PlayerStatus {
  health: number;
  ammo: number;
  tag_id: string;
  display_name: string;
  is_connected: boolean;
  event_type: string;
  last_active: number; // Date.now() timestamp in milliseconds
}

export interface GameStatus {
  players: Record<string, PlayerStatus>;
  event_type: string;
}

// incoming event from a player
export interface PlayerTagged {
  tag_id: string;
  image_ids: string[];
  event_type: string;
}

// // Server-enriched version with additional details
// export interface PlayerTaggedEnriched extends PlayerTagged {
//   healthpoints: number;
//   server_info?: string; //notes for humans to debug
// }

export interface ImageInfo {
  image_id: string;
  device_id: string;
  timestamp: number;
  file_location: string;
}


export interface GameServerStats {
  server_info: {
    uptime_seconds: number;
    start_time: string;
  };
  activity: {
    total_images: number;
    recent_images: ImageInfo[];
  };
  game_state: {
    active_players: number;
    players: Record<string, PlayerStatus>;
  };
}

export interface ServerMetrics {
  queue: {
    size: number;
    processing: boolean;
  };
  timestamp: number;
}

// Custom request interface with user ID header
export interface GameRequest extends Request {
  headers: Request["headers"] & {
    "x-device-id"?: string;
  };
}
