import { Router, Request, Response } from "express";
import { logger } from "../utils/logger";

const router: Router = Router();

interface SystemStatus {
  server: {
    status: "RUNNING" | "STARTING" | "STOPPING";
    uptime: string;
    startTime: string;
    environment: string;
    version: string;
    port: string | number;
  };
  system: {
    platform: string;
    nodeVersion: string;
    pid: number;
    memory: {
      used: string;
      total: string;
      percentage: string;
    };
    cpu: NodeJS.CpuUsage;
  };
  metrics: {
    requestCount: number;
    errorCount: number;
    averageResponseTime: string;
  };
}

// Simple in-memory metrics (in production, use proper metrics collection)
let requestCount = 0;
let errorCount = 0;
let totalResponseTime = 0;
const startTime = new Date();

// Middleware to track metrics
export const metricsMiddleware = (
  _req: Request,
  res: Response,
  next: Function,
) => {
  const start = Date.now();
  requestCount++;

  res.on("finish", () => {
    const duration = Date.now() - start;
    totalResponseTime += duration;

    if (res.statusCode >= 400) {
      errorCount++;
    }
  });

  next();
};

const formatBytes = (bytes: number): string => {
  const sizes = ["Bytes", "KB", "MB", "GB"];
  if (bytes === 0) return "0 Bytes";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
};

const formatUptime = (seconds: number): string => {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  return `${days}d ${hours}h ${minutes}m ${secs}s`;
};

router.get("/", (req: Request, res: Response) => {
  try {
    const memoryUsage = process.memoryUsage();
    const totalMemory = memoryUsage.heapTotal + memoryUsage.external;
    const usedMemory = memoryUsage.heapUsed;
    const memoryPercentage = ((usedMemory / totalMemory) * 100).toFixed(2);

    const status: SystemStatus = {
      server: {
        status: "RUNNING",
        uptime: formatUptime(process.uptime()),
        startTime: startTime.toISOString(),
        environment: process.env.NODE_ENV || "development",
        version: "1.0.0",
        port: process.env.PORT || 3000,
      },
      system: {
        platform: process.platform,
        nodeVersion: process.version,
        pid: process.pid,
        memory: {
          used: formatBytes(usedMemory),
          total: formatBytes(totalMemory),
          percentage: `${memoryPercentage}%`,
        },
        cpu: process.cpuUsage(),
      },
      metrics: {
        requestCount,
        errorCount,
        averageResponseTime:
          requestCount > 0
            ? `${(totalResponseTime / requestCount).toFixed(2)}ms`
            : "0ms",
      },
    };

    logger.info("System status requested", { ip: req.ip });
    res.status(200).json(status);
  } catch (error) {
    logger.error("Status check failed", { error, ip: req.ip });
    res.status(500).json({
      error: "Failed to get system status",
      timestamp: new Date().toISOString(),
    });
  }
});

export { router as statusRouter };
