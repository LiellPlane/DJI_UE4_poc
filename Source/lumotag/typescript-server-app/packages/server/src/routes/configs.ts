


import * as fs from 'fs';
import * as path from 'path';

export interface DeviceInfo {
  tag_id: string;
  display_name: string;
}

// Load device to tag ID mapping from JSON config file
const configPath = path.join(__dirname, '../config/device-tag-mapping.json');
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

export function getDeviceInfo(deviceId: string): DeviceInfo {
  return {
    tag_id: config.device_ids[deviceId].tag_id,
    display_name: config.device_ids[deviceId].display_name
  };
}
