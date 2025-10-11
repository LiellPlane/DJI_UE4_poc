/* eslint-disable @typescript-eslint/no-unused-vars */
import express, { Response, Router } from "express";
// import sharp from 'sharp'; // Removed for simplicity
// import { LRUCache } from 'lru-cache'; // Removed for simplicity
import { logger } from "../utils/logger";
import { imageSaver } from "./image-saver";
import { validateAndExtractDeviceId } from "../utils/request-helpers";
import {
  GameRequest,
  PlayerStatus,
  GameStatus,
  PlayerTagged,
  ImageInfo,
  GameServerStats,
  UploadRequest,
  ServerMetrics,
  ReqKillScreenResponse,
  KillShot,
} from "../types";
import { getDeviceInfo, DeviceInfo } from "./configs"
import * as dgram from "dgram";

const router: Router = express.Router();

// Simple background image processing queue
interface QueuedImage {
  imageId: string;
  imageBuffer: Buffer; // Store processed buffer instead of base64 string
  device_id: string;
  timestamp: number;
}

const imageQueue: QueuedImage[] = [];
let isProcessing = false;


// Background thread for cyclic operations
let backgroundThread: NodeJS.Timeout | null = null;

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
        device_id: queuedImage.device_id,
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
  InGamePlayers: Record<string, PlayerStatus>; //mapping device id to display id and tag id 
  imagesReceived: ImageInfo[]; //when a device uploads an image, during trigger event or tag event
  killShots: Record<string, PlayerTagged>; //when a players health reaches zero - put the details here so we can find the corresponding killshot
  startTime: number;
  lastHealTime: number; // Track when all players were last healed
}

// Initial state
const createInitialGameState = (): GameState => ({
  InGamePlayers: {},
  imagesReceived: [],
  killShots: {},
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
      active_players: Object.keys(state.InGamePlayers).length,
      players: { ...state.InGamePlayers },
    },
  }),
};

// Background thread operations
const runBackgroundOperations = (): void => {
  try {
    // Run healing
    healPlayers();
    
    // Update connection status for all players
    updateConnectionStatus_all_players();
    
    // Process image queue
    processImageQueue();
    
    logger.debug('Background operations completed');
  } catch (error) {
    logger.error('CRITICAL: Background thread error - crashing server:', error);
    process.exit(1); // Crash the entire server
  }
};

// Start background thread
const startBackgroundThread = (): void => {
  if (backgroundThread) {
    logger.warn('Background thread already running');
    return;
  }
  
  backgroundThread = setInterval(runBackgroundOperations, 500); // Run every 500ms
  logger.info('Background thread started (500ms interval)');
};

// Stop background thread
const stopBackgroundThread = (): void => {
  if (backgroundThread) {
    clearInterval(backgroundThread);
    backgroundThread = null;
    logger.info('Background thread stopped');
  }
};

// Start thread immediately when module loads
startBackgroundThread();

// Direct state management - no thin wrapper class needed
let gameState = createInitialGameState();

// Direct functions for state management
const addImageInfo = (info: ImageInfo): void => {
  gameState = gameLogic.addImageInfo(gameState, info);
};

const updateLastSeen = (deviceid: string): void => {
  const player = gameState.InGamePlayers[deviceid];
  if (player) {
    gameState.InGamePlayers[deviceid] = {
      ...player,
      last_active: Date.now()
    };
  }
}

const getDeviceIdByTagId = (tag_id: string): string | undefined => {
  return Object.keys(gameState.InGamePlayers).find(
    deviceId => gameState.InGamePlayers[deviceId].tag_id === tag_id
  );
};

const tagPlayer = (tag_id: string): PlayerStatus | null => {
  const deviceId = getDeviceIdByTagId(tag_id);
  if (deviceId) {
    const player = gameState.InGamePlayers[deviceId];
    const updatedPlayer = {
      ...player,
      health: player.health - 15,
      last_active: Date.now()
    };
    gameState.InGamePlayers[deviceId] = updatedPlayer;
    return updatedPlayer;
  } else {
    logger.warn(`Tag ID not found: ${tag_id}`);
    return null;
  }
}

const eliminatePlayer = (tag_id: string): PlayerStatus | null => {
  const deviceId = getDeviceIdByTagId(tag_id);
  if (deviceId) {
    const player = gameState.InGamePlayers[deviceId];
    const updatedPlayer = {
      ...player,
      isEliminated: true,
      last_active: Date.now()
    };
    gameState.InGamePlayers[deviceId] = updatedPlayer;
    return updatedPlayer;
  } else {
    logger.warn(`Tag ID not found: ${tag_id}`);
    return null;
  }
}

const updateConnectionStatus_all_players = (): void => {
  const now = Date.now();
  const connectionTimeoutMs = 1000; // 2 seconds
  
  Object.keys(gameState.InGamePlayers).forEach(deviceId => {
    const player = gameState.InGamePlayers[deviceId];
    const isConnected = (now - player.last_active) <= connectionTimeoutMs;
    
    gameState.InGamePlayers[deviceId] = {
      ...player,
      is_connected: isConnected
    };
  });
}

const getStats = (): GameServerStats => {
  return gameLogic.getStats(gameState);
};


// Performance middleware - add request timing
router.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    // Skip performance logging for noisy endpoints
    if (req.path.includes('/images/upload') || 
        req.path.includes('/gamestate') ||
        req.path.includes('/stats') ||
        req.path.includes('/metrics')) {
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
  Object.entries(gameState.InGamePlayers).forEach(([tagId, player]) => {
    if (player.health < 100 && !player.isEliminated) {
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
    InGamePlayers: updatedPlayers,
    lastHealTime: currentTime,
  };
};

// GAMESTATE endpoint - GET /api/v1/gamestate (for devices)
router.get("/gamestate", (req: GameRequest, res: Response) => {
  try {
    // Validate required header
    const deviceId = validateAndExtractDeviceId(req, res);
    if (!deviceId) return; // Response already sent by validation function
    
    // logger.debug(`deviceId: ${deviceId}`);
    
    // Fast O(1) lookup - if user doesn't exist, create new PlayerStatus
    if (!gameState.InGamePlayers[deviceId]) {
      const deviceMapping = getDeviceInfo(deviceId);
      
      if (!deviceMapping) {
        // Device not in config - reject request
        logger.error(`Unknown device_id: ${deviceId}`);
        return res.status(400).json({ 
          error: "Unknown device_id", 
          device_id: deviceId,
          message: "Device not found in configuration" 
        });
      }
      
      // Device found in config - create new player with mapped values
      const newPlayer: PlayerStatus = {
        health: 50,
        ammo: 0,
        tag_id: deviceMapping.tag_id,
        display_name: deviceMapping.display_name,
        is_connected: true,
        event_type: "PlayerStatus",
        last_active: Date.now(),
        isEliminated: false,
      };
      gameState.InGamePlayers[deviceId] = newPlayer;
      logger.info(`Created new player: ${deviceId} with display_name: ${deviceMapping.display_name}`);
    }
    
    // Update last seen for this specific device
    updateLastSeen(deviceId);
    
    // Return current game state directly
    const gameUpdate: GameStatus = {
      players: gameState.InGamePlayers,
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

// DASHBOARD GAMESTATE endpoint - GET /api/v1/dashboard/gamestate (no auth required)
router.get("/dashboard/gamestate", (_req: GameRequest, res: Response) => {
  try {
    // Return current game state directly - no device validation needed
    const gameUpdate: GameStatus = {
      players: gameState.InGamePlayers,
      event_type: "GameStatus",
    };

    res.set({
      "Content-Type": "application/json",
      "Cache-Control": "no-cache",
    });

    return res.json(gameUpdate);
  } catch (error) {
    logger.error("Dashboard gamestate error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// IMAGE UPLOAD endpoint - POST /api/v1/images/upload
router.post("/images/upload", (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  
  try {
    // Validate required header
    const deviceId = validateAndExtractDeviceId(req, res);
    if (!deviceId) return; // Response already sent by validation function
    
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
      device_id: deviceId,
      timestamp: Date.now(),
    });

    // Start processing queue in background (non-blocking)
    processImageQueue().catch(error => {
      logger.error("Queue processing error:", error);
    });

    logger.debug(`[${timestamp}] Image queued for processing:
    ID: ${uploadRequest.image_id}
    deviceId: ${deviceId}
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
router.post("/events", async (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  
  try {
    // Validate required header
    const deviceId = validateAndExtractDeviceId(req, res);
    if (!deviceId) return; // Response already sent by validation function

    const eventType = req.body.event_type;

    let taggedEvent: PlayerTagged;

    switch (eventType) {

      case 'PlayerTagged':
        taggedEvent = {
          ...req.body
        } as PlayerTagged;
        

        logger.info(`[${timestamp}] PLAYER TAGGED - deviceId: ${deviceId}, tag_ids: ${taggedEvent.tag_ids.join(',')}, Event: ${JSON.stringify(taggedEvent)}`);

        // Process each tagged player
        for (const tag_id of taggedEvent.tag_ids) {
          const taggedPlayer: PlayerStatus | null = tagPlayer(tag_id);

          if (taggedPlayer && taggedPlayer.health <= 0 && !taggedPlayer.isEliminated) {
            const eliminatedDeviceId = getDeviceIdByTagId(tag_id);
            if (eliminatedDeviceId) {
              eliminatePlayer(tag_id);
              gameState.killShots[eliminatedDeviceId] = taggedEvent;
              logger.info(`[${timestamp}] 🔍 DEBUG: Storing killshot for ${eliminatedDeviceId} with ${taggedEvent.image_ids.length} images: ${JSON.stringify(taggedEvent.image_ids)}`);
              logger.info(`Player eliminated: ${taggedPlayer.display_name} (${eliminatedDeviceId}) by device ${deviceId}`);
            }
          }
        }
    

        break;

      case 'ReqKillScreen':
        const killShotEvent = {
          ...req.body
        } as KillShot;

        logger.info(`[${timestamp}] KILLSHOT REQUEST - deviceId: ${deviceId}, Event: ${JSON.stringify(killShotEvent)}`);

        try {
          const killScreenResponse = await processKillScreenRequest(deviceId, timestamp);
          return res.json(killScreenResponse);
        } catch (error) {
          logger.error(`[${timestamp}] KillShot processing error:`, error);
          if (error instanceof Error && error.message.includes('not found')) {
            return res.status(404).json({ 
              error: "Not found", 
              details: error.message,
              device_id: deviceId 
            });
          }
          return res.status(500).json({ error: "Internal server error" });
        }

        
      default:
        throw new Error(`Unknown event type: ${eventType}. Supported types: PlayerTagged, KillShot`);
    }

    // Event processed successfully - no storage needed

    logger.info(`[${timestamp}] Event received successfully:
    Type: ${eventType}
    deviceId: ${deviceId}
    Parsed Data: ${JSON.stringify(taggedEvent)}`);

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
  // logger.info(`[${timestamp}] 📊 Stats request`);

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


// Helper function to process killscreen requests (fail-fast - no retries)
async function processKillScreenRequest(deviceId: string, timestamp: string): Promise<ReqKillScreenResponse> {
  logger.info(`[${timestamp}] KILLSCREEN REQUEST - deviceId: ${deviceId}`);

  // Look up eliminated player by device ID
  const eliminatedPlayer = gameState.killShots[deviceId];
  if (!eliminatedPlayer) {
    logger.warn(`[${timestamp}] Eliminated player not found for device: ${deviceId}`);
    throw new Error(`Eliminated player not found for device: ${deviceId}`);
  }

  // Get the tagger's display name (who eliminated this player)
  const playerData = gameState.InGamePlayers[deviceId];
  const displayNameTagger = playerData.display_name;

  // Fail-fast: Try to get images once, no retry loop
  const imageIds = eliminatedPlayer.image_ids;
  const imageDatas: string[] = [];

  logger.info(`[${timestamp}] Attempting to retrieve ${imageIds.length} images: ${imageIds.join(', ')}`);

  for (const imageId of imageIds) {
    const imageData = await imageSaver.getImageAsBase64(imageId);
    
    if (imageData) {
      logger.debug(`[${timestamp}] Image ${imageId} found (${imageData.length} bytes)`);
    } else {
      logger.warn(`[${timestamp}] Image ${imageId} not found - failing fast`);
    }
    
    imageDatas.push(imageData!);
  }

  // Create response
  const killScreenResponse: ReqKillScreenResponse = {
    display_name_tagger: displayNameTagger,
    image_datas: imageDatas,
    event_type: "ReqKillScreenResponse"
  };

  logger.info(`[${timestamp}] Killscreen response sent - ${imageDatas.length}/${imageIds.length} images for device: ${deviceId}`);
  return killScreenResponse;
}


// RESET endpoint - POST /api/v1/reset (simple endpoint to reset all game state and images)
router.post("/reset", async (_req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  
  try {
    logger.info(`[${timestamp}] 🔄 RESET REQUEST - Clearing all game state and images`);

    // Clear all game state
    gameState = createInitialGameState();
    
    // Clear image queue
    imageQueue.length = 0;
    
    // Clear all images from disk
    const cleanupResult = await imageSaver.cleanupAllImages();
    
    logger.info(`[${timestamp}] ✅ RESET COMPLETE - Cleared ${Object.keys(gameState.InGamePlayers).length} players, ${cleanupResult.deletedCount} images`);
    
    return res.json({
      status: "success",
      message: "Game state and images reset successfully",
      cleared: {
        players: 0, // Will always be 0 after reset
        images: cleanupResult.deletedCount,
        queue_size: 0
      },
      errors: cleanupResult.errors,
      timestamp: Date.now()
    });
    
  } catch (error) {
    logger.error(`[${timestamp}] Reset error:`, error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// TEST UDP BROADCAST endpoint - POST /api/v1/test/udp-broadcast
router.post("/test/udp-broadcast", (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  
  try {
    const { tag_ids, image_ids } = req.body;
    
    if (!tag_ids || !Array.isArray(tag_ids) || tag_ids.length === 0) {
      return res.status(400).json({ error: "tag_ids array is required" });
    }
    
    // Create PlayerTagged event (matches Python Pydantic model)
    const event: PlayerTagged = {
      tag_ids,
      image_ids: image_ids || [],
      event_type: "PlayerTagged"
    };
    
    logger.info(`[${timestamp}] 📡 UDP BROADCAST TEST - Broadcasting tag event for tag_ids: ${tag_ids.join(',')}`);
    
    // Broadcast UDP
    const sock = dgram.createSocket('udp4');
    const message = Buffer.from(JSON.stringify(event));
    
    // Must bind before setting broadcast option
    sock.bind(() => {
      sock.setBroadcast(true);
      
      // Broadcast to 255.255.255.255 (limited broadcast)
      sock.send(message, 5000, '255.255.255.255', (err) => {
        if (err) {
          logger.error(`[${timestamp}] UDP broadcast error:`, err);
        } else {
          logger.info(`[${timestamp}] ✅ UDP broadcast sent successfully`);
        }
        sock.close();
      });
    });
    
    return res.json({
      status: "success",
      message: "UDP broadcast sent",
      event,
      timestamp: Date.now()
    });
    
  } catch (error) {
    logger.error(`[${timestamp}] UDP broadcast test error:`, error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Cleanup function for graceful shutdown
const cleanup = (): void => {
  stopBackgroundThread();
};

// Handle process termination
process.on('SIGTERM', cleanup);
process.on('SIGINT', cleanup);

export { router as gameRouter, stopBackgroundThread };
