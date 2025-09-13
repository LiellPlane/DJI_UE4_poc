import { promises as fs } from 'fs';
import path from 'path';
import { logger } from '../utils/logger';

class ImageSaver {
  private uploadsDir: string;

  constructor() {
    // Use __dirname to get the current file's directory, then go up to server root
    // This ensures images are saved in packages/server/uploads/ regardless of where you run the server from
    this.uploadsDir = path.join(__dirname, '..', '..', 'uploads');
    this.ensureUploadsDir();
  }

  private async ensureUploadsDir(): Promise<void> {
    try {
      await fs.mkdir(this.uploadsDir, { recursive: true });
    } catch (error) {
      logger.error('Failed to create uploads directory:', error);
    }
  }

  async saveImage(imageId: string, base64Data: string): Promise<{
    filename: string;
    filepath: string;
    size: number;
  }> {
    // Decode base64 to buffer
    const imageBuffer = Buffer.from(base64Data, 'base64');
    
    // Generate unique filename
    // const timestamp = Date.now();
    const filename = `${imageId}.jpg`;
    const filepath = path.join(this.uploadsDir, filename);
    
    // Save asynchronously (non-blocking)
    await fs.writeFile(filepath, imageBuffer);
    
    return {
      filename,
      filepath,
      size: imageBuffer.length
    };
  }

  async getImageStats(): Promise<{
    totalImages: number;
    totalSize: number;
  }> {
    try {
      const files = await fs.readdir(this.uploadsDir);
      const imageFiles = files.filter(f => f.endsWith('.jpg'));
      
      let totalSize = 0;
      for (const file of imageFiles) {
        const stats = await fs.stat(path.join(this.uploadsDir, file));
        totalSize += stats.size;
      }
      
      return {
        totalImages: imageFiles.length,
        totalSize
      };
    } catch (error) {
      return { totalImages: 0, totalSize: 0 };
    }
  }

  async cleanupAllImages(): Promise<{
    deletedCount: number;
    errors: string[];
  }> {
    const errors: string[] = [];
    let deletedCount = 0;
    
    try {
      // Ensure directory exists first
      await this.ensureUploadsDir();
      
      const files = await fs.readdir(this.uploadsDir);
      const imageFiles = files.filter(f => f.endsWith('.jpg'));
      
      logger.info(`🧹 Cleaning up ${imageFiles.length} image files from uploads directory`);
      
      for (const file of imageFiles) {
        try {
          await fs.unlink(path.join(this.uploadsDir, file));
          deletedCount++;
        } catch (error) {
          errors.push(`Failed to delete ${file}: ${error}`);
          logger.warn(`Failed to delete image file ${file}:`, error);
        }
      }
      
      logger.info(`✅ Cleanup complete: ${deletedCount} files deleted, ${errors.length} errors`);
      
      return { deletedCount, errors };
    } catch (error) {
      const errorMsg = `Failed to read uploads directory: ${error}`;
      errors.push(errorMsg);
      logger.error('Image cleanup failed:', error);
      return { deletedCount, errors };
    }
  }
}

export const imageSaver = new ImageSaver();
