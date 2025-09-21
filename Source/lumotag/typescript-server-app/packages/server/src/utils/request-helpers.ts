import { Request, Response } from "express";

export function extractDeviceId(req: Request): string {
  return req.headers["x-device-id"] as string;
}

export function validateAndExtractDeviceId(req: Request, res: Response): string | null {
  const deviceID = extractDeviceId(req);
  
  if (!deviceID || deviceID === 'undefined') {
    res.status(400).json({ 
      error: "Bad Request", 
      message: "Missing required header: x-device-id",
      required_headers: ["x-device-id"]
    });
    return null;
  }
  
  return deviceID;
}