import express, { Response, Router } from "express";
import { z } from "zod";
// import sharp from 'sharp'; // Removed for simplicity
// import { LRUCache } from 'lru-cache'; // Removed for simplicity
import { logger } from "../utils/logger";
import { imageSaver } from "./image-saver";
import {
  GameRequest,
  PlayerStatus,
  GameUpdate,
  ImageInfo,
  EventInfo,
  GameServerStats,
} from "../types";

const router: Router = express.Router();

// High-performance in-memory storage
class GameServerState {
  private static instance: GameServerState;
  private startTime: number = Date.now();

  // Simplified - no image storage, just metadata tracking

  // Recent activity tracking
  public imagesReceived: ImageInfo[] = [];
  public eventsReceived: EventInfo[] = [];

  // Dynamic player data (simulating your Python server's behavior)
  public playersData: Record<string, PlayerStatus> = {
    testself: {
      health: 100,
      ammo: 30,
      tag_id: "testself",
      display_name: "tinytim",
      event_type: "PlayerStatus", // REQUIRED field
    },
    player_002: {
      health: 85,
      ammo: 22,
      tag_id: "player_002",
      display_name: "mongo",
      event_type: "PlayerStatus", // REQUIRED field
    },
    player_003: {
      health: 95,
      ammo: 18,
      tag_id: "player_003",
      display_name: "dildort",
      event_type: "PlayerStatus", // REQUIRED field
    },
  };

  private gamestateCounter = 0;

  static getInstance(): GameServerState {
    if (!GameServerState.instance) {
      GameServerState.instance = new GameServerState();
    }
    return GameServerState.instance;
  }

  // Image storage methods removed for simplicity

  addImageInfo(info: ImageInfo): void {
    this.imagesReceived.push(info);
    // Keep only last 100 entries for memory efficiency
    if (this.imagesReceived.length > 100) {
      this.imagesReceived = this.imagesReceived.slice(-100);
    }
  }

  addEventInfo(info: EventInfo): void {
    this.eventsReceived.push(info);
    // Keep only last 100 entries for memory efficiency
    if (this.eventsReceived.length > 100) {
      this.eventsReceived = this.eventsReceived.slice(-100);
    }
  }

  // Simulate dynamic gamestate like your Python server
  updateGameState(): GameUpdate {
    const currentTime = Date.now() / 1000;

    // Update player stats dynamically (matching your Python logic)
    Object.entries(this.playersData).forEach(([tagId, player]) => {
      if (tagId === "testself") {
        // Make testself health jiggle for testing
        const baseHealth = 75;
        const sineVariation = Math.floor(20 * Math.sin(currentTime * 2));
        const randomJitter = Math.floor(5 * Math.sin(currentTime * 7));
        const newHealth = Math.max(
          20,
          Math.min(100, baseHealth + sineVariation + randomJitter),
        );

        this.playersData[tagId] = {
          ...player,
          health: newHealth,
          event_type: "PlayerStatus", // Ensure event_type is always present
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

        this.playersData[tagId] = {
          ...player,
          health: newHealth,
          event_type: "PlayerStatus", // Ensure event_type is always present
        };
      }

      // Simulate ammo consumption
      const ammoConsumed = Math.floor(currentTime / 5) % 5;
      const baseAmmo =
        tagId === "testself" ? 30 : tagId === "player_002" ? 22 : 18;
      const newAmmo = Math.max(0, baseAmmo - ammoConsumed);

      this.playersData[tagId] = {
        ...this.playersData[tagId],
        ammo: newAmmo,
        event_type: "PlayerStatus", // Ensure event_type is always present
      };
    });

    this.gamestateCounter++;

    // Log occasionally to reduce spam (every 10th request)
    if (this.gamestateCounter % 10 === 0) {
      logger.info(
        `🎮 Gamestate request #${this.gamestateCounter} - returning ${Object.keys(this.playersData).length} players`,
      );
    }

    return {
      players: { ...this.playersData },
      event_type: "GameUpdate", // REQUIRED field
    };
  }

  getStats(): GameServerStats {
    return {
      server_info: {
        uptime_seconds: (Date.now() - this.startTime) / 1000,
        start_time: new Date(this.startTime).toISOString(),
      },
      activity: {
        total_images: this.imagesReceived.length,
        total_events: this.eventsReceived.length,
        recent_images: this.imagesReceived.slice(-5),
        recent_events: this.eventsReceived.slice(-5),
      },
      game_state: {
        active_players: Object.keys(this.playersData).length,
        players: { ...this.playersData },
      },
    };
  }
}

// Validation schemas using Zod - event_type is REQUIRED
const uploadRequestSchema = z.object({
  image_id: z.string().min(1),
  image_data: z.string().min(1), // base64 string
  event_type: z.string().default("UploadRequest"), // REQUIRED with default
  timestamp: z.number().optional(),
});

const eventSchema = z
  .object({
    event_type: z.string().min(1), // REQUIRED - must be provided
  })
  .passthrough(); // Allow additional fields

// Get singleton instance
const gameState = GameServerState.getInstance();

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
    // Fast validation using Zod - will add default event_type if missing
    const validatedData = uploadRequestSchema.parse(req.body);

    // Simple validation - just check if we have image data
    if (!validatedData.image_data || validatedData.image_data.length === 0) {
      logger.warn(
        `[${timestamp}] ❌ No image data for image ${validatedData.image_id}`,
      );
      return res.status(400).json({ error: "No image data provided" });
    }

    // Save image to disk (async, non-blocking)
    const savedImage = await imageSaver.saveImage(
      validatedData.image_id,
      validatedData.image_data
    );

    // Store image metadata with actual file info
    const imageInfo: ImageInfo = {
      image_id: validatedData.image_id,
      user_id: userId,
      timestamp: validatedData.timestamp || Date.now(),
      size_bytes: savedImage.size,
      received_at: Date.now(),
    };

    gameState.addImageInfo(imageInfo);

    logger.info(`[${timestamp}] 📸 Image saved successfully:
    🔍 ID: ${validatedData.image_id}
    👤 User: ${userId}
    📏 Size: ${savedImage.size.toLocaleString()} bytes
    💾 File: ${savedImage.filename}`);

    // Simulate small processing delay (like your Python server)
    await new Promise((resolve) => setTimeout(resolve, 10));

    return res.json({
      status: "success",
      image_id: validatedData.image_id,
      filename: savedImage.filename,
      size_bytes: savedImage.size,
      processed_at: Date.now(),
      message: "Image saved successfully",
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      logger.warn(`[${timestamp}] ❌ Validation error:`, error.errors);
      return res.status(400).json({
        error: "Validation failed",
        details: error.errors,
      });
    }

    logger.error(`[${timestamp}] ❌ Image upload error:`, error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// EVENTS endpoint - POST /api/v1/events
router.post("/events", (req: GameRequest, res: Response) => {
  const timestamp = new Date().toISOString().slice(11, 23);
  const userId = req.headers["x-user-id"] || "unknown";

  try {
    // Fast validation - event_type is REQUIRED
    const validatedData = eventSchema.parse(req.body);

    // Store event info
    const eventInfo: EventInfo = {
      event_type: validatedData.event_type,
      user_id: userId,
      data: { ...validatedData },
      received_at: Date.now(),
    };

    gameState.addEventInfo(eventInfo);

    logger.info(`[${timestamp}] 📨 Event received successfully:
    🏷️  Type: ${validatedData.event_type}
    👤 User: ${userId}
    📦 Data: ${JSON.stringify(validatedData)}`);

    return res.json({
      status: "success",
      event_type: validatedData.event_type,
      processed_at: Date.now(),
      message: "Event processed successfully",
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      logger.warn(`[${timestamp}] ❌ Event validation error:`, error.errors);
      return res.status(400).json({
        error: "Validation failed",
        details: error.errors,
      });
    }

    logger.error(`[${timestamp}] ❌ Event processing error:`, error);
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
