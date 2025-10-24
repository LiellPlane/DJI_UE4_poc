// Game server types - matching the server types exactly
export interface PlayerStatus {
  health: number;
  ammo: number;
  tag_id: string;
  display_name: string;
  is_connected: boolean;
  event_type: string;
  last_active: number; // Date.now() timestamp in milliseconds
  isEliminated: boolean;
}

export interface GameStatus {
  players: Record<string, PlayerStatus>;
  event_type: string;
}

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
  image_storage?: {
    total_files: number;
    total_size_bytes: number;
    total_size_mb: number;
  };
}

export interface ServerMetrics {
  queue: {
    size: number;
    processing: boolean;
    peak_size?: number; // Max queue size since last check
    total_processed?: number; // Total images processed
  };
  timestamp: number;
}

export interface ReqKillScreenResponse {
  display_name_tagger: string;
  image_datas: string[]; // Array of base64 encoded JPEG image data for HTTP transmission
  event_type: string;
}

export interface KillShot {
  // when a device sees that it has zero health - request the kill images from the server
  image_data: string; // base64 encoded JPEG
  device_id: string;
  display_name: string;
  event_type: string;
}