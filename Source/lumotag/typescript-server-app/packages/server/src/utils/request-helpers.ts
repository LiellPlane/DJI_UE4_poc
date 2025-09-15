import { Request } from "express";

/**
 * Extracts the user ID from the X-User-ID header
 * @param req Express request object
 * @returns The user ID from the header
 */
export function extractUserId(req: Request): string {
  return req.headers["x-user-id"] as string;
}