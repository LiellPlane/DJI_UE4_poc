import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { logger } from '../utils/logger';

const router: Router = Router();

interface HealthStatus {
  status: 'UP' | 'DOWN';
  timestamp: string;
  uptime: number;
  memory: NodeJS.MemoryUsage;
  environment: string;
  version: string;
}

// Health check schema for validation
const healthCheckSchema = z.object({
  detailed: z.string().optional(),
});

router.get('/', (req: Request, res: Response) => {
  try {
    // Validate query parameters
    const { detailed } = healthCheckSchema.parse(req.query);
    
    const healthStatus: HealthStatus = {
      status: 'UP',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      environment: process.env.NODE_ENV || 'development',
      version: '1.0.0',
    };

    // Add detailed information if requested
    if (detailed === 'true') {
      const detailedStatus = {
        ...healthStatus,
        cpu: process.cpuUsage(),
        platform: process.platform,
        nodeVersion: process.version,
        pid: process.pid,
      };
      
      logger.info('Detailed health check requested', { ip: req.ip });
      return res.status(200).json(detailedStatus);
    }
    
    return res.status(200).json(healthStatus);
  } catch (error) {
    logger.error('Health check failed', { error, ip: req.ip });
    return res.status(503).json({
      status: 'DOWN',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
    });
  }
});

// Readiness probe
router.get('/ready', (_req: Request, res: Response) => {
  // Add any readiness checks here (database connectivity, etc.)
  res.status(200).json({
    status: 'READY',
    timestamp: new Date().toISOString(),
  });
});

// Liveness probe
router.get('/live', (_req: Request, res: Response) => {
  res.status(200).json({
    status: 'ALIVE',
    timestamp: new Date().toISOString(),
  });
});

export { router as healthRouter };
