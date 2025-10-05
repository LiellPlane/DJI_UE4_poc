import { Request, Response, NextFunction } from "express";
import { logger } from "../utils/logger";

interface AppError extends Error {
  statusCode?: number;
  isOperational?: boolean;
}

export const errorHandler = (
  err: AppError,
  req: Request,
  res: Response,
  _next: NextFunction,
): void => {
  // Default error values
  let statusCode = 500;
  let message = "Internal Server Error";
  let details: any = undefined;

  // Handle custom app errors
  if (err.statusCode) {
    statusCode = err.statusCode;
    message = err.message;
  }

  // Log error
  logger.error("Request error", {
    error: err.message,
    stack: err.stack,
    statusCode,
    method: req.method,
    url: req.originalUrl,
    ip: req.ip,
    userAgent: req.get("User-Agent"),
  });

  // Send error response
  const errorResponse: any = {
    error: message,
    timestamp: new Date().toISOString(),
    path: req.originalUrl,
    method: req.method,
  };

  if (details) {
    errorResponse.details = details;
  }

  // Include stack trace in development
  if (process.env.NODE_ENV === "development") {
    errorResponse.stack = err.stack;
  }

  res.status(statusCode).json(errorResponse);
};

// Async error handler wrapper
export const asyncHandler =
  (fn: Function) => (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
