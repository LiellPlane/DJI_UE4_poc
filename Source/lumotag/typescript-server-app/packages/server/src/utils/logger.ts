import winston from "winston";

// Default to 'info' for good visibility of operations (uploads, events, etc.)
// Change to 'warn' for minimal logging, 'debug' for verbose output
// Override with LOG_LEVEL env var if needed
const logLevel = process.env.LOG_LEVEL || "info";
const nodeEnv = process.env.NODE_ENV || "development";

// Custom log format (simplified for better performance)
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
  winston.format.errors({ stack: true }),
  winston.format.json(),
  // Removed prettyPrint() - it's expensive and not needed for file logs
);

// Console format for development - clean and compact
const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({ format: "HH:mm:ss" }),
  winston.format.printf(({ timestamp, level, message }) => {
    // Simple format: timestamp [level]: message (no JSON metadata spam)
    return `${timestamp} [${level}]: ${message}`;
  }),
);

// Create logger
export const logger = winston.createLogger({
  level: logLevel,
  format: logFormat,
  defaultMeta: { service: "typescript-server" },
  transports: [
    // File transport for all logs
    new winston.transports.File({
      filename: "logs/error.log",
      level: "error",
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    new winston.transports.File({
      filename: "logs/combined.log",
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
  ],
});

// Console transport for development
if (nodeEnv === "development") {
  logger.add(
    new winston.transports.Console({
      format: consoleFormat,
    }),
  );
} else {
  // Structured logging for production
  logger.add(
    new winston.transports.Console({
      format: winston.format.json(),
    }),
  );
}

// Handle exceptions and rejections
logger.exceptions.handle(
  new winston.transports.File({ filename: "logs/exceptions.log" }),
);

logger.rejections.handle(
  new winston.transports.File({ filename: "logs/rejections.log" }),
);
