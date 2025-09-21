import { Request, Response } from "express";

export function extractDeviceId(req: Request): string {
  return req.headers["X-device-ID"] as string;
}

export function validateAndExtractDeviceId(req: Request, res: Response): string | null {
  const deviceID = extractDeviceId(req);
  
  if (!deviceID || deviceID === 'undefined') {
    res.status(400).json({ 
      error: "Bad Request", 
      message: "Missing required header: X-device-ID",
      required_headers: ["X-device-ID"]
    });
    return null;
  }
  
  return deviceID;
}