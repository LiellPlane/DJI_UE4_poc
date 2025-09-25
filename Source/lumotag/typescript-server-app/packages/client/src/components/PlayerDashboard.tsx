import React from "react";
import { useQuery } from "react-query";
import { Users, Heart, Zap, Wifi, WifiOff, Clock, Skull } from "lucide-react";
import { apiService } from "@/services/api";
import type { GameStatus, PlayerStatus } from "@/types";

export const PlayerDashboard: React.FC = () => {
  const {
    data: gameState,
    isLoading,
    error,
    isError,
  } = useQuery<GameStatus>("gameState", apiService.getGameState, {
    refetchInterval: 2000, // Update every 2 seconds for player status
  });

  const formatTimeSince = (timestamp: number): string => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  const getHealthColor = (health: number): string => {
    if (health >= 75) return "health-good";
    if (health >= 50) return "health-medium";
    if (health >= 25) return "health-low";
    return "health-critical";
  };

  const getConnectionStatus = (player: PlayerStatus): string => {
    if (player.isEliminated) return "eliminated";
    return player.is_connected ? "connected" : "disconnected";
  };

  if (isLoading) {
    return (
      <div className="dashboard-card loading">
        <div className="loading-spinner"></div>
        <p>Loading players...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="dashboard-card error">
        <div className="card-header">
          <Users className="icon error" />
          <h2>Player Dashboard</h2>
        </div>
        <div className="error-message">
          Failed to fetch player data
          <p className="error-details">{error?.toString()}</p>
        </div>
      </div>
    );
  }

  const players = gameState?.players || {};
  const playerCount = Object.keys(players).length;

  return (
    <div className="dashboard-card">
      <div className="card-header">
        <Users className="icon" />
        <h2>Players ({playerCount})</h2>
      </div>

      <div className="card-content">
        {playerCount === 0 ? (
          <div className="empty-state">
            <Users className="empty-icon" />
            <p>No active players</p>
          </div>
        ) : (
          <div className="players-grid">
            {Object.entries(players).map(([deviceId, player]) => (
              <div key={deviceId} className={`player-card ${getConnectionStatus(player)}`}>
                <div className="player-header">
                  <div className="player-name">
                    <span className="display-name">{player.display_name}</span>
                    <span className="tag-id">#{player.tag_id}</span>
                  </div>
                  <div className="connection-status">
                    {player.isEliminated ? (
                      <Skull className="icon eliminated" />
                    ) : player.is_connected ? (
                      <Wifi className="icon connected" />
                    ) : (
                      <WifiOff className="icon disconnected" />
                    )}
                  </div>
                </div>

                <div className="player-stats">
                  <div className="stat">
                    <Heart className={`stat-icon ${getHealthColor(player.health)}`} />
                    <div className="stat-info">
                      <span className="stat-label">Health</span>
                      <span className={`stat-value ${getHealthColor(player.health)}`}>
                        {player.health}%
                      </span>
                    </div>
                  </div>

                  <div className="stat">
                    <Zap className="stat-icon" />
                    <div className="stat-info">
                      <span className="stat-label">Ammo</span>
                      <span className="stat-value">{player.ammo}</span>
                    </div>
                  </div>

                  <div className="stat">
                    <Skull className={`stat-icon ${player.isEliminated ? 'eliminated' : 'alive'}`} />
                    <div className="stat-info">
                      <span className="stat-label">Status</span>
                      <span className={`stat-value ${player.isEliminated ? 'eliminated' : 'alive'}`}>
                        {player.isEliminated ? 'ELIMINATED' : 'ALIVE'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="player-footer">
                  <Clock className="footer-icon" />
                  <span className="last-active">
                    {formatTimeSince(player.last_active)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
