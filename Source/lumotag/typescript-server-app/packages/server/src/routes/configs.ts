
import { Router, Response } from "express";

import * as fs from 'fs';
import * as path from 'path';

export interface DeviceInfo {
  tag_id: string;
  display_name: string;
}

// Load device to tag ID mapping from JSON config file in project root
const configPath = path.join(process.cwd(), '../../device-tag-mapping.json');

let config: any;
try {
  console.log(`🔍 Loading config from: ${configPath}`);
  
  if (!fs.existsSync(configPath)) {
    console.error(`❌ CONFIG ERROR: File not found at ${configPath}`);
    console.error(`   Current working directory: ${process.cwd()}`);
    console.error(`   Expected config file: device-tag-mapping.json`);
    console.error(`   Please ensure the config file exists in the typescript-server-app root directory`);
    process.exit(1);
  }
  
  const configData = fs.readFileSync(configPath, 'utf8');
  config = JSON.parse(configData);
  
  if (!config.device_ids) {
    console.error(`❌ CONFIG ERROR: Missing 'device_ids' section in ${configPath}`);
    process.exit(1);
  }
  
  console.log(`✅ Config loaded successfully with ${Object.keys(config.device_ids).length} device(s)`);
} catch (error) {
  console.error(`❌ CONFIG ERROR: Failed to load ${configPath}`);
  console.error(`   Error: ${error instanceof Error ? error.message : String(error)}`);
  console.error(`   Current working directory: ${process.cwd()}`);
  process.exit(1);
}

export function getDeviceInfo(deviceId: string): DeviceInfo | null {
  if (!config.device_ids[deviceId]) {
    console.warn(`⚠️  Unknown device_id '${deviceId}' (available: ${Object.keys(config.device_ids).join(', ')})`);
    return null;
  }
  
  return {
    tag_id: config.device_ids[deviceId].tag_id,
    display_name: config.device_ids[deviceId].display_name
  };
}

const router: Router = Router();

router.get("/device-mapping", (_req, res: Response) => {
  res.json(config);
});

export { router as configRouter };
