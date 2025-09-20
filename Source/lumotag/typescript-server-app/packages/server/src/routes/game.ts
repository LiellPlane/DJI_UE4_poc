/* eslint-disable @typescript-eslint/no-unused-vars */
import express, { Response, Router } from "express";
// import sharp from 'sharp'; // Removed for simplicity
// import { LRUCache } from 'lru-cache'; // Removed for simplicity
import { logger } from "../utils/logger";
import { imageSaver } from "./image-saver";
import { extractUserId, validateAndExtractUserId } from "../utils/request-helpers";
import {
  GameRequest,
  PlayerStatus,
  GameStatus,
  PlayerTagged,
  ImageInfo,
  GameServerStats,
  UploadRequest,
  ServerMetrics,
} from "../types";

const router: Router = express.Router();

// Simple background image processing queue
interface QueuedImage {
  imageId: string;
  imageBuffer: Buffer; // Store processed buffer instead of base64 string
  userId: string;
  timestamp: number;
}

const imageQueue: QueuedImage[] = [];
let isProcessing = false;

// Background processor - processes one image at a time
async function processImageQueue() {
  if (isProcessing || imageQueue.length === 0) return;
  
  isProcessing = true;
  
  while (imageQueue.length > 0) {
    const queuedImage = imageQueue.shift()!;
    
    try {
      const savedImage = await imageSaver.saveImageBuffer(
        queuedImage.imageId,
        queuedImage.imageBuffer
      );
      
      // Store image metadata after successful save
      const imageInfo: ImageInfo = {
        image_id: queuedImage.imageId,
        user_id: queuedImage.userId,
        timestamp: queuedImage.timestamp,
        file_location: savedImage.filename,
      };
      
      addImageInfo(imageInfo);
      
      logger.debug(`Background: Image ${queuedImage.imageId} saved successfully (${savedImage.size.toLocaleString()} bytes)`);
    } catch (error) {
      logger.error(`Background: Failed to save image ${queuedImage.imageId}:`, error);
    }
  }
  
  isProcessing = false;
}

// Pure data structure (no methods)
interface GameState {
  playersData: Record<string, PlayerStatus>;
  imagesReceived: ImageInfo[];
  gamestateCounter: number;
  startTime: number;
  lastHealTime: number; // Track when all players were last healed
}

// Initial state
const createInitialGameState = (): GameState => ({
  playersData: {
    // dummy data for sanity check
    // testself: {
    //   health: 75,
    //   ammo: 30,
    //   tag_id: "testself",
    //   display_name: "tinytim",
    //   event_type: "PlayerStatus",
    // },
    // player_002: {
    //   health: 85,
    //   ammo: 22,
    //   tag_id: "player_002",
    //   display_name: "mongo",
    //   event_type: "PlayerStatus",
    // },
    // player_003: {
    //   health: 95,
    //   ammo: 18,
    //   tag_id: "player_003",
    //   display_name: "dildort",
    //   event_type: "PlayerStatus",
    // },
  },
  imagesReceived: [],
  gamestateCounter: 0,
  startTime: Date.now(),
  lastHealTime: Date.now(),
});

// Pure functions for game logic
const gameLogic = {
  addImageInfo: (state: GameState, info: ImageInfo): GameState => ({
    ...state, // this spaffs all the members(?) into this scope of {}. Then we can ovveride the members just by using their 
    // names. Yes its very weird and disgusting, and potentially retarded
    imagesReceived: [...state.imagesReceived, info].slice(-100), // Keep only last 100
    //implicitly return the whole thing in {} which will repackage it as an object
  }),


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

// Direct state management - no thin wrapper class needed
let gameState = createInitialGameState();

// Direct functions for state management
const addImageInfo = (info: ImageInfo): void => {
  gameState = gameLogic.addImageInfo(gameState, info);
};



const getStats = (): GameServerStats => {
  return gameLogic.getStats(gameState);
};

// For debugging/testing
const getState = (): GameState => {
  return gameState; // Return immutable reference
};

// Performance middleware - add request timing
router.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    // Skip performance logging for upload and gamestate endpoints
    if (req.path.includes('/images/upload') || req.path.includes('/gamestate')) {
      return;
    }
    
    const duration = Number(process.hrtime.bigint() - start) / 1e6; // Convert to milliseconds
    
    // Different thresholds for different endpoints
    let threshold = 50; // Default: 50ms for most requests
    
    if (duration > threshold) {
      logger.warn(
        `Slow request: ${req.method} ${req.path} took ${duration.toFixed(2)}ms (threshold: ${threshold}ms)`,
      );
    }
  });
  next();
});

// Dedicated healing function
const healPlayers = (): void => {
  const currentTime = Date.now();
  const timeSinceLastHeal = currentTime - gameState.lastHealTime;
  const shouldHeal = timeSinceLastHeal >= 5000; // 5000ms = 5 seconds
  
  if (!shouldHeal) return;
  
  const updatedPlayers: Record<string, PlayerStatus> = {};
  
  // Apply healing to all players
  Object.entries(gameState.playersData).forEach(([tagId, player]) => {
    if (player.health < 100) {
      const oldHealth = player.health;
      const newHealth = Math.min(100, player.health + 1);
      logger.info(`HEALING: ${tagId} (${player.display_name}) ${oldHealth} -> ${newHealth}`);
      
      updatedPlayers[tagId] = {
        ...player,
        health: newHealth,
      };
    } else {
      updatedPlayers[tagId] = { ...player };
    }
  });
  
  // Update the game state with healed players
  gameState = {
    ...gameState,
    playersData: updatedPlayers,
    lastHealTime: currentTime,
  };
};

// GAMESTATE endpoint - GET /api/v1/gamestate
router.get("/gamestate", (req: GameRequest, res: Response) => {
  try {
    // Validate required header
    const userId = validateAndExtractUserId(req, res);
    if (!userId) return; // Response already sent by validation function
    
    logger.debug(`User ID: ${userId}`);
    
    // Fast O(1) lookup - if user doesn't exist, create new PlayerStatus
    if (!gameState.playersData[userId]) {
      const newPlayer: PlayerStatus = {
        health: 50,
        ammo: 0,
        tag_id: Math.random().toString(36).substr(2, 9), // Random 9-character string
        display_name: userId,
        event_type: "PlayerStatus",
      };
      gameState.playersData[userId] = newPlayer;
      logger.info(`Created new player: ${userId} with display_name: ${userId}`);
    }
    
    // Call dedicated healing function
    healPlayers();
    // Return current game state directly
    const gameUpdate: GameStatus = {
      players: gameState.playersData,
      event_type: "GameStatus",
    };

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
router.post("/images/upload", (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  
  try {
    // Validate required header
    const userId = validateAndExtractUserId(req, res);
    if (!userId) return; // Response already sent by validation function
    
    // Try to parse body as UploadRequest - crash if it doesn't work
    const uploadRequest: UploadRequest = {
      ...req.body
    };

    // Convert base64 to buffer immediately to save memory
    const imageBuffer = Buffer.from(uploadRequest.image_data, 'base64');
    
    // Add image to background processing queue (no more base64 string!)
    imageQueue.push({
      imageId: uploadRequest.image_id,
      imageBuffer: imageBuffer,
      userId: userId,
      timestamp: Date.now(),
    });

    // Start processing queue in background (non-blocking)
    processImageQueue().catch(error => {
      logger.error("Queue processing error:", error);
    });

    logger.debug(`[${timestamp}] Image queued for processing:
    ID: ${uploadRequest.image_id}
    User: ${userId}
    Queue length: ${imageQueue.length}`);

    // Return immediately - don't wait for image to save
    return res.json({
      status: "success",
      image_id: uploadRequest.image_id,
      event_type: uploadRequest.event_type,
      processed_at: Date.now(),
      message: "Image queued for processing",
    });
  } catch (error) {
    logger.error(`[${timestamp}] Image upload error:`, error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// EVENTS endpoint - POST /api/v1/events
router.post("/events", (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  
  try {
    // Validate required header
    const userId = validateAndExtractUserId(req, res);
    if (!userId) return; // Response already sent by validation function

    const eventType = req.body.event_type;

    let parsedEvent: PlayerTagged;

    switch (eventType) {

      case 'PlayerTagged':
        parsedEvent = {
          ...req.body
        } as PlayerTagged;
        logger.info(`[${timestamp}] PLAYER TAGGED - User: ${userId}, Event: ${JSON.stringify(parsedEvent)}`);
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
    const gameStats = getStats();
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

// METRICS endpoint - GET /metrics (simple queue monitoring)
router.get("/metrics", (_req: GameRequest, res: Response) => {
  try {
    const metrics: ServerMetrics = {
      queue: {
        size: imageQueue.length,
        processing: isProcessing,
      },
      timestamp: Date.now(),
    };
    
    return res.json(metrics);
  } catch (error) {
    logger.error("Metrics error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export { router as gameRouter };
