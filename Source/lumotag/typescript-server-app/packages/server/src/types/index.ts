// Types for the game server
import { Request } from "express";

// Game-specific types matching your Python Pydantic models exactly
// These match the structures in lumotag_events.py
// All fields are REQUIRED - no defaults, crash if missing
export interface UploadRequest {
  image_id: string;
  image_data: string; // base64 encoded JPEG
  event_type: string; // REQUIRED - no default
}

export interface PlayerStatus {
  health: number;
  ammo: number;
  tag_id: string;
  display_name: string;
  event_type: string; // REQUIRED - no default
}

export interface GameUpdate {
  players: Record<string, PlayerStatus>;
  event_type: string; // REQUIRED - no default
}

export interface PlayerTagged {
  tag_id: string;
  image_ids: string[];
  event_type: string; // REQUIRED - no default
}

export interface ImageInfo {
  image_id: string;
  user_id: string;
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

// Custom request interface with user ID header
export interface GameRequest extends Request {
  headers: Request["headers"] & {
    "x-user-id"?: string;
  };
}
