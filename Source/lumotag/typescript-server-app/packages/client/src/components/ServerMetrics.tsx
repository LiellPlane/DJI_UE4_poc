import React from "react";
import { useQuery } from "react-query";
import { Server, Clock, HardDrive, Activity, AlertCircle } from "lucide-react";
import { apiService } from "@/services/api";
import type { GameServerStats, ServerMetrics } from "@/types";

export const ServerMetricsComponent: React.FC = () => {
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
  } = useQuery<GameServerStats>("serverStats", apiService.getStats, {
    refetchInterval: 5000, // Update every 5 seconds
  });

  const {
    data: metrics,
    isLoading: metricsLoading,
    error: metricsError,
  } = useQuery<ServerMetrics>("serverMetrics", apiService.getMetrics, {
    refetchInterval: 3000, // Update every 3 seconds for queue status
  });

  const formatUptime = (seconds: number): string => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 MB";
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const isLoading = statsLoading || metricsLoading;
  const isError = statsError || metricsError;

  if (isLoading) {
    return (
      <div className="dashboard-card loading">
        <div className="loading-spinner"></div>
        <p>Loading server metrics...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="dashboard-card error">
        <div className="card-header">
          <Server className="icon error" />
          <h2>Server Metrics</h2>
        </div>
        <div className="error-message">
          Failed to fetch server metrics
          <p className="error-details">
            {statsError?.toString() || metricsError?.toString()}
          </p>
        </div>
      </div>
    );
  }

  const queueWarning = metrics && metrics.queue.size > 5;
  const processingStatus = metrics?.queue.processing ? "Processing" : "Idle";

  return (
    <div className="dashboard-card">
      <div className="card-header">
        <Server className="icon" />
        <h2>Server Metrics</h2>
        {queueWarning && <AlertCircle className="icon warning" />}
      </div>

      <div className="card-content">
        <div className="metrics-grid">
          {/* Server Uptime */}
          <div className="metric-item">
            <Clock className="metric-icon" />
            <div className="metric-info">
              <span className="metric-label">Uptime</span>
              <span className="metric-value">
                {stats ? formatUptime(stats.server_info.uptime_seconds) : "N/A"}
              </span>
            </div>
          </div>

          {/* Active Players */}
          <div className="metric-item">
            <Activity className="metric-icon" />
            <div className="metric-info">
              <span className="metric-label">Active Players</span>
              <span className="metric-value">
                {stats?.game_state.active_players || 0}
              </span>
            </div>
          </div>

          {/* Image Storage */}
          {stats?.image_storage && (
            <div className="metric-item">
              <HardDrive className="metric-icon" />
              <div className="metric-info">
                <span className="metric-label">Storage Used</span>
                <span className="metric-value">
                  {formatBytes(stats.image_storage.total_size_bytes)}
                </span>
              </div>
            </div>
          )}

          {/* Processing Queue */}
          <div className="metric-item">
            <Activity className={`metric-icon ${metrics?.queue.processing ? 'processing' : ''}`} />
            <div className="metric-info">
              <span className="metric-label">Queue Status</span>
              <span className={`metric-value ${queueWarning ? 'warning' : ''}`}>
                Current: {metrics?.queue.size || 0} | Peak: {metrics?.queue.peak_size || 0}
              </span>
              <span className="metric-sublabel">
                {processingStatus} | Total: {metrics?.queue.total_processed || 0} processed
              </span>
            </div>
          </div>
        </div>

        {/* Server Start Time */}
        <div className="server-info">
          <span className="info-label">Started:</span>
          <span className="info-value">
            {stats ? new Date(stats.server_info.start_time).toLocaleString() : "N/A"}
          </span>
        </div>

        {queueWarning && (
          <div className="warning-banner">
            <AlertCircle className="warning-icon" />
            <span>Image processing queue is backed up ({metrics?.queue.size} items)</span>
          </div>
        )}
      </div>
    </div>
  );
};
