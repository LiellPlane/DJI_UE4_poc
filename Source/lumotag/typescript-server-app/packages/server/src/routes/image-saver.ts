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
    return this.saveImageBuffer(imageId, imageBuffer);
  }

  async saveImageBuffer(imageId: string, imageBuffer: Buffer): Promise<{
    filename: string;
    filepath: string;
    size: number;
  }> {
    // Generate unique filename
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

  async getImageAsBase64(imageId: string): Promise<string | null> {
    const filename = `${imageId}.jpg`;
    const filepath = path.join(this.uploadsDir, filename);
    
    try {
      // Check if file exists - will throw if not
      await fs.access(filepath);
      
      // Read file and convert to base64 - will throw if fails
      const imageBuffer = await fs.readFile(filepath);
      return imageBuffer.toString('base64');
    } catch (error) {
      // Return null for retry logic - the caller will decide to crash after timeout
      return null;
    }
  }

  async pruneOldImages(): Promise<void> {
    try {
      const files = await fs.readdir(this.uploadsDir);
      const imageFiles = files.filter(f => f.endsWith('.jpg'));
      
      if (imageFiles.length <= 1000) return;
      
      // Get file stats with timestamps
      const filesWithStats = await Promise.all(
        imageFiles.map(async (file) => {
          const filepath = path.join(this.uploadsDir, file);
          const stats = await fs.stat(filepath);
          return { file, mtime: stats.mtime.getTime() };
        })
      );
      
      // Sort by modification time (oldest first)
      filesWithStats.sort((a, b) => a.mtime - b.mtime);
      
      // Delete oldest 10
      const toDelete = filesWithStats.slice(0, 500);
      
      for (const { file } of toDelete) {
        await fs.unlink(path.join(this.uploadsDir, file));
      }
      
      logger.info(`Pruned ${toDelete.length} oldest images (total: ${imageFiles.length})`);
    } catch (error) {
      logger.error('Image pruning failed:', error);
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
