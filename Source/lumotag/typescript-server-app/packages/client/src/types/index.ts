export interface HealthStatus {
  status: 'UP' | 'DOWN';
  timestamp: string;
  uptime: number;
  memory: NodeJS.MemoryUsage;
  environment: string;
  version: string;
}

export interface SystemStatus {
  server: {
    status: 'RUNNING' | 'STARTING' | 'STOPPING';
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
