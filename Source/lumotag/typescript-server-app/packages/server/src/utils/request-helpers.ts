import { Request, Response } from "express";

/**
 * Extracts the user ID from the X-User-ID header
 * @param req Express request object
 * @returns The user ID from the header
 */
export function extractUserId(req: Request): string {
  return req.headers["x-device-id"] as string;
}

/**
 * Validates that the X-User-ID header is present and valid
 * @param req Express request object
 * @param res Express response object
 * @returns The user ID if valid, or sends 400 error and returns null
 */
export function validateAndExtractUserId(req: Request, res: Response): string | null {
  const userId = extractUserId(req);
  
  if (!userId || userId === 'undefined') {
    res.status(400).json({ 
      error: "Bad Request", 
      message: "Missing required header: X-User-ID",
      required_headers: ["X-User-ID"]
    });
    return null;
  }
  
  return userId;
}