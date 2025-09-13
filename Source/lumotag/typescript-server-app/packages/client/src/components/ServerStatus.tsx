import React from "react";
import { useQuery } from "react-query";
import { Activity, Clock, Cpu, HardDrive } from "lucide-react";
import { apiService } from "@/services/api";
import type { HealthStatus } from "@/types";

export const ServerStatus: React.FC = () => {
  const {
    data: status,
    isLoading,
    error,
    isError,
  } = useQuery<HealthStatus>("serverStatus", apiService.getHealth, {
    refetchInterval: 5000,
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
    const sizes = ["Bytes", "KB", "MB", "GB"];
    if (bytes === 0) return "0 Bytes";
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
  };

  if (isLoading) {
    return (
      <div className="status-card loading">
        <div className="loading-spinner"></div>
        <p>Loading server status...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="status-card error">
        <div className="status-header">
          <Activity className="icon error" />
          <h2>Server Status</h2>
        </div>
        <div className="error-message">
          Failed to fetch server status
          <p className="error-details">{error?.toString()}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="status-card">
      <div className="status-header">
        <Activity className={`icon ${status?.status.toLowerCase()}`} />
        <h2>Server Status</h2>
      </div>

      <div className="status-content">
        <div className={`status-badge ${status?.status.toLowerCase()}`}>
          {status?.status}
        </div>

        <div className="status-grid">
          <div className="status-item">
            <Clock className="item-icon" />
            <div>
              <span className="label">Uptime</span>
              <span className="value">{formatUptime(status?.uptime || 0)}</span>
            </div>
          </div>

          <div className="status-item">
            <HardDrive className="item-icon" />
            <div>
              <span className="label">Memory Used</span>
              <span className="value">
                {formatBytes(status?.memory?.heapUsed || 0)}
              </span>
            </div>
          </div>

          <div className="status-item">
            <Cpu className="item-icon" />
            <div>
              <span className="label">Environment</span>
              <span className="value">{status?.environment}</span>
            </div>
          </div>

          <div className="status-item">
            <Activity className="item-icon" />
            <div>
              <span className="label">Last Updated</span>
              <span className="value">
                {status?.timestamp
                  ? new Date(status.timestamp).toLocaleTimeString()
                  : "N/A"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
