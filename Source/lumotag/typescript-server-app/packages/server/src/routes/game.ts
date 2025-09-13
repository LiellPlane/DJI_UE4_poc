import express, { Response, Router } from "express";
// import sharp from 'sharp'; // Removed for simplicity
// import { LRUCache } from 'lru-cache'; // Removed for simplicity
import { logger } from "../utils/logger";
import { imageSaver } from "./image-saver";
import {
  GameRequest,
  PlayerStatus,
  GameUpdate,
  PlayerTagged,
  ImageInfo,
  GameServerStats,
  UploadRequest,
} from "../types";

const router: Router = express.Router();

// Pure data structure (no methods)
interface GameState {
  playersData: Record<string, PlayerStatus>;
  imagesReceived: ImageInfo[];
  gamestateCounter: number;
  startTime: number;
}

// Initial state
const createInitialGameState = (): GameState => ({
  playersData: {
    testself: {
      health: 100,
      ammo: 30,
      tag_id: "testself",
      display_name: "tinytim",
      event_type: "PlayerStatus",
    },
    player_002: {
      health: 85,
      ammo: 22,
      tag_id: "player_002",
      display_name: "mongo",
      event_type: "PlayerStatus",
    },
    player_003: {
      health: 95,
      ammo: 18,
      tag_id: "player_003",
      display_name: "dildort",
      event_type: "PlayerStatus",
    },
  },
  imagesReceived: [],
  gamestateCounter: 0,
  startTime: Date.now(),
});

// Pure functions for game logic
const gameLogic = {
  addImageInfo: (state: GameState, info: ImageInfo): GameState => ({
    ...state,
    imagesReceived: [...state.imagesReceived, info].slice(-100), // Keep only last 100
  }),

  updateGameState: (state: GameState): { newState: GameState; gameUpdate: GameUpdate } => {
    const currentTime = Date.now() / 1000;
    const updatedPlayers: Record<string, PlayerStatus> = {};

    // Update player stats dynamically (matching your Python logic)
    Object.entries(state.playersData).forEach(([tagId, player]) => {
      if (tagId === "testself") {
        // Make testself health jiggle for testing
        const baseHealth = 75;
        const sineVariation = Math.floor(20 * Math.sin(currentTime * 2));
        const randomJitter = Math.floor(5 * Math.sin(currentTime * 7));
        const newHealth = Math.max(
          20,
          Math.min(100, baseHealth + sineVariation + randomJitter),
        );

        updatedPlayers[tagId] = {
          ...player,
          health: newHealth,
          event_type: "PlayerStatus",
        };
      } else {
        const baseHealth = tagId === "player_002" ? 85 : 95;
        const healthVariation = Math.floor(
          10 * (0.5 - (currentTime % 10) / 20),
        );
        const newHealth = Math.max(
          10,
          Math.min(100, baseHealth + healthVariation),
        );

        updatedPlayers[tagId] = {
          ...player,
          health: newHealth,
          event_type: "PlayerStatus",
        };
      }

      // Simulate ammo consumption
      const ammoConsumed = Math.floor(currentTime / 5) % 5;
      const baseAmmo =
        tagId === "testself" ? 30 : tagId === "player_002" ? 22 : 18;
      const newAmmo = Math.max(0, baseAmmo - ammoConsumed);

      updatedPlayers[tagId] = {
        ...updatedPlayers[tagId],
        ammo: newAmmo,
        event_type: "PlayerStatus",
      };
    });

    const newCounter = state.gamestateCounter + 1;

    // Log occasionally to reduce spam (every 10th request)
    if (newCounter % 10 === 0) {
      logger.info(
        `🎮 Gamestate request #${newCounter} - returning ${Object.keys(updatedPlayers).length} players`,
      );
    }

    const newState: GameState = {
      ...state,
      playersData: updatedPlayers,
      gamestateCounter: newCounter,
    };

    const gameUpdate: GameUpdate = {
      players: { ...updatedPlayers },
      event_type: "GameUpdate",
    };

    return { newState, gameUpdate };
  },

  getStats: (state: GameState): GameServerStats => ({
    server_info: {
      uptime_seconds: (Date.now() - state.startTime) / 1000,
      start_time: new Date(state.startTime).toISOString(),
    },
    activity: {
      total_images: state.imagesReceived.length,
      recent_images: state.imagesReceived.slice(-5),
    },
    game_state: {
      active_players: Object.keys(state.playersData).length,
      players: { ...state.playersData },
    },
  }),
};

// Thin wrapper for persistence (replaces singleton class)
class GameStateManager {
  private state: GameState;

  constructor() {
    this.state = createInitialGameState();
  }

  addImageInfo(info: ImageInfo): void {
    this.state = gameLogic.addImageInfo(this.state, info);
  }

  updateGameState(): GameUpdate {
    const { newState, gameUpdate } = gameLogic.updateGameState(this.state);
    this.state = newState;
    return gameUpdate;
  }

  getStats(): GameServerStats {
    return gameLogic.getStats(this.state);
  }

  // For debugging/testing
  getState(): GameState {
    return { ...this.state }; // Return copy to prevent external mutation
  }
}

// Simple validation like Python server - no complex schemas

// Create single instance (replaces singleton pattern)
const gameState = new GameStateManager();

// Performance middleware - add request timing
router.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    const duration = Number(process.hrtime.bigint() - start) / 1e6; // Convert to milliseconds
    
    // Different thresholds for different endpoints
    let threshold = 50; // Default: 50ms for most requests
    if (req.path.includes('/images/upload')) {
      threshold = 100; // 100ms for image uploads (includes disk I/O)
    } else if (req.path.includes('/gamestate')) {
      threshold = 25; // 25ms for gamestate (should be fast)
    }
    
    if (duration > threshold) {
      logger.warn(
        `Slow request: ${req.method} ${req.path} took ${duration.toFixed(2)}ms (threshold: ${threshold}ms)`,
      );
    }
  });
  next();
});

// GAMESTATE endpoint - GET /api/v1/gamestate
router.get("/gamestate", (_req: GameRequest, res: Response) => {
  try {
    const gameUpdate = gameState.updateGameState();

    res.set({
      "Content-Type": "application/json",
      "Cache-Control": "no-cache",
    });

    return res.json(gameUpdate);
  } catch (error) {
    logger.error("Gamestate error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// IMAGE UPLOAD endpoint - POST /api/v1/images/upload
router.post("/images/upload", async (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  const userId = req.headers["x-user-id"] || "unknown";

  try {
    // Try to parse body as UploadRequest - crash if it doesn't work
    // Use spread operator (like Python **) to unpack all fields
    const uploadRequest: UploadRequest = {
      ...req.body
    };

    // Save image to disk (async, non-blocking)
    const savedImage = await imageSaver.saveImage(
      uploadRequest.image_id,
      uploadRequest.image_data
    );

    // Store image metadata with actual file info
    const imageInfo: ImageInfo = {
      image_id: uploadRequest.image_id,
      user_id: userId,
      timestamp: Date.now(),
      file_location: savedImage.filename,
    };

    gameState.addImageInfo(imageInfo);

    logger.info(`[${timestamp}] Image saved successfully:
    ID: ${uploadRequest.image_id}
    User: ${userId}
    Size: ${savedImage.size.toLocaleString()} bytes
    File: ${savedImage.filename}`);

    // Simulate small processing delay (like your Python server)
    await new Promise((resolve) => setTimeout(resolve, 10));

    return res.json({
      status: "success",
      image_id: uploadRequest.image_id,
      event_type: uploadRequest.event_type,
      filename: savedImage.filename,
      processed_at: Date.now(),
      message: "Image saved successfully",
    });
  } catch (error) {
    logger.error(`[${timestamp}] Image upload error:`, error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// EVENTS endpoint - POST /api/v1/events
router.post("/events", (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  const userId = req.headers["x-user-id"] || "unknown";

  try {

    const eventType = req.body.event_type;

    let parsedEvent: PlayerTagged;

    switch (eventType) {

      case 'PlayerTagged':
        parsedEvent = {
          ...req.body
        } as PlayerTagged;
        break;

      default:
        throw new Error(`Unknown event type: ${eventType}. Supported types: PlayerTagged`);
    }

    // Event processed successfully - no storage needed

    logger.info(`[${timestamp}] Event received successfully:
    Type: ${eventType}
    User: ${userId}
    Parsed Data: ${JSON.stringify(parsedEvent)}`);

    return res.json({
      status: "success",
      event_type: eventType,
      processed_at: Date.now(),
      message: "Event processed successfully",
    });
  } catch (error) {
    logger.error(`[${timestamp}] Event processing error:`, error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// STATS endpoint - GET /stats
router.get("/stats", async (_req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  logger.info(`[${timestamp}] 📊 Stats request`);

  try {
    const gameStats = gameState.getStats();
    const imageStats = await imageSaver.getImageStats();
    
    const stats = {
      ...gameStats,
      image_storage: {
        total_files: imageStats.totalImages,
        total_size_bytes: imageStats.totalSize,
        total_size_mb: Math.round(imageStats.totalSize / (1024 * 1024) * 100) / 100
      }
    };
    
    return res.json(stats);
  } catch (error) {
    logger.error("Stats error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export { router as gameRouter };
