import React from "react";
import { useQuery } from "react-query";
import { Camera, Clock, User, Image as ImageIcon } from "lucide-react";
import { apiService } from "@/services/api";
import type { GameServerStats, ImageInfo } from "@/types";

export const ImageActivity: React.FC = () => {
  const {
    data: stats,
    isLoading,
    error,
    isError,
  } = useQuery<GameServerStats>("serverStats", apiService.getStats, {
    refetchInterval: 3000, // Update every 3 seconds for recent activity
  });

  const formatTimeSince = (timestamp: number): string => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  const formatImageId = (imageId: string): string => {
    // Truncate long image IDs for display
    if (imageId.length > 20) {
      return `${imageId.substring(0, 10)}...${imageId.substring(imageId.length - 7)}`;
    }
    return imageId;
  };

  const formatDeviceId = (deviceId: string): string => {
    // Show last 8 characters of device ID
    if (deviceId.length > 8) {
      return `...${deviceId.substring(deviceId.length - 8)}`;
    }
    return deviceId;
  };

  if (isLoading) {
    return (
      <div className="dashboard-card loading">
        <div className="loading-spinner"></div>
        <p>Loading image activity...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="dashboard-card error">
        <div className="card-header">
          <Camera className="icon error" />
          <h2>Image Activity</h2>
        </div>
        <div className="error-message">
          Failed to fetch image activity
          <p className="error-details">{error?.toString()}</p>
        </div>
      </div>
    );
  }

  const recentImages = stats?.activity.recent_images || [];
  const totalImages = stats?.activity.total_images || 0;

  return (
    <div className="dashboard-card">
      <div className="card-header">
        <Camera className="icon" />
        <h2>Image Activity</h2>
        <div className="header-stats">
          <span className="total-count">{totalImages} total</span>
        </div>
      </div>

      <div className="card-content">
        {recentImages.length === 0 ? (
          <div className="empty-state">
            <ImageIcon className="empty-icon" />
            <p>No recent image uploads</p>
          </div>
        ) : (
          <div className="activity-list">
            <div className="activity-header">
              <span className="header-title">Recent Uploads ({recentImages.length})</span>
            </div>
            
            {recentImages.map((image: ImageInfo) => (
              <div key={image.image_id} className="activity-item">
                <div className="activity-icon">
                  <Camera className="item-icon" />
                </div>
                
                <div className="activity-details">
                  <div className="activity-main">
                    <span className="image-id">{formatImageId(image.image_id)}</span>
                    <span className="device-info">
                      <User className="inline-icon" />
                      {formatDeviceId(image.device_id)}
                    </span>
                  </div>
                  
                  <div className="activity-meta">
                    <Clock className="meta-icon" />
                    <span className="timestamp">{formatTimeSince(image.timestamp)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Summary stats */}
        {stats?.image_storage && (
          <div className="activity-summary">
            <div className="summary-item">
              <span className="summary-label">Stored Files:</span>
              <span className="summary-value">{stats.image_storage.total_files}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Storage Used:</span>
              <span className="summary-value">{stats.image_storage.total_size_mb.toFixed(1)} MB</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
