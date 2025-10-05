import express, { Application, Request, Response, NextFunction } from "express";
import cors from "cors";
import compression from "compression";
import morgan from "morgan";
import dotenv from "dotenv";
import path from "path";
import { logger } from "./utils/logger";
import { errorHandler } from "./middlewares/errorHandler";
import { statusRouter } from "./routes/status";
import { gameRouter } from "./routes/game";
import { imageSaver } from "./routes/image-saver";
import { configRouter } from "./routes/configs";

// Load environment variables
dotenv.config();

const app: Application = express();
const PORT = parseInt(process.env.PORT || '8080', 10);


app.use(
  cors({
    origin: "*", // Allow all origins for LAN operation
  }),
);

// Performance middleware
app.use(compression());

// Logging middleware
app.use(
  morgan("combined", { 
    stream: { write: (msg) => logger.info(msg.trim()) },
    skip: (req) => {
      // Skip logging for noisy endpoints
      return req.originalUrl.includes('/images/upload') || 
             req.originalUrl.includes('/gamestate') ||
             req.originalUrl.includes('/stats') ||
             req.originalUrl.includes('/metrics');
    }
  }),
);

// Body parsing middleware
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// Avatar files
app.use('/avatars', express.static(path.join(__dirname, '..', 'public', 'avatars')));

// Request logging
app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  res.on("finish", () => {
    // Skip logging for noisy endpoints
    if (req.originalUrl.includes('/images/upload') || 
        req.originalUrl.includes('/gamestate') ||
        req.originalUrl.includes('/stats') ||
        req.originalUrl.includes('/metrics')) {
      return;
    }
    
    const duration = Date.now() - start;
    logger.info(`${req.method} ${req.originalUrl}`, {
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
      userAgent: req.get("User-Agent"),
    });
  });
  next();
});

// API Routes
app.use("/api/status", statusRouter);
app.use("/api/v1", gameRouter);
app.use("/api/v1", configRouter);

// Root endpoint
app.get("/", (_req: Request, res: Response) => {
  res.json({
    message: "TypeScript Game Server API",
    version: "1.0.0",
    timestamp: new Date().toISOString(),
    endpoints: {
      status: "/api/status",
      gamestate: "/api/v1/gamestate",
      images_upload: "/api/v1/images/upload",
      events: "/api/v1/events",
      stats: "/api/v1/stats",
    },
  });
});

// Error handling middleware (must be last)
app.use(errorHandler);

// 404 handler
app.use("*", (req: Request, res: Response) => {
  logger.warn(`Route not found: ${req.method} ${req.originalUrl}`, {
    ip: req.ip,
  });
  res.status(404).json({
    error: "Route not found",
    message: `Cannot ${req.method} ${req.originalUrl}`,
    timestamp: new Date().toISOString(),
  });
});

// Cleanup images on server start
const cleanupImages = async () => {
  try {
    const result = await imageSaver.cleanupAllImages();
    if (result.deletedCount > 0) {
      logger.info(`🧹 Startup cleanup: Removed ${result.deletedCount} old image files`);
    }
    if (result.errors.length > 0) {
      logger.warn(`⚠️ Cleanup had ${result.errors.length} errors`);
    }
  } catch (error) {
    logger.error('Failed to cleanup images on startup:', error);
  }
};

// Start server
const HOST = process.env.HOST || '0.0.0.0';
const server = app.listen(PORT, HOST, () => {
  logger.info(`Server running on ${HOST}:${PORT}`, {
    environment: process.env.NODE_ENV || "development",
    host: HOST,
    port: PORT,
  });
});

// Run cleanup after server is ready
server.on('listening', async () => {
  await cleanupImages();
});

// Graceful shutdown
const gracefulShutdown = (signal: string) => {
  logger.info(`${signal} received, shutting down gracefully`);
  server.close(() => {
    logger.info("Server closed");
    process.exit(0);
  });
};

process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

// Handle unhandled promise rejections
process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection at:", promise, "reason:", reason);
});

// Handle uncaught exceptions
process.on("uncaughtException", (error) => {
  logger.error("Uncaught Exception:", error);
  process.exit(1);
});

export default app;
