import React from 'react';
import { useQuery } from 'react-query';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Server, TrendingUp, AlertTriangle } from 'lucide-react';
import { apiService } from '@/services/api';
import type { SystemStatus } from '@/types';

export const SystemMetrics: React.FC = () => {
  const { data: metrics, isLoading, error, isError } = useQuery<SystemStatus>(
    'systemMetrics',
    apiService.getStatus,
    {
      refetchInterval: 10000, // Refetch every 10 seconds
    }
  );

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  const memoryData = metrics?.system.memory ? [
    {
      name: 'Used Memory',
      value: parseFloat(metrics.system.memory.percentage),
      color: '#FF6B6B',
    },
    {
      name: 'Free Memory',
      value: 100 - parseFloat(metrics.system.memory.percentage),
      color: '#4ECDC4',
    },
  ] : [];

  const requestData = [
    { name: 'Total Requests', value: metrics?.metrics.requestCount || 0 },
    { name: 'Errors', value: metrics?.metrics.errorCount || 0 },
    { name: 'Success Rate', value: metrics ? ((metrics.metrics.requestCount - metrics.metrics.errorCount) / Math.max(metrics.metrics.requestCount, 1)) * 100 : 0 },
  ];

  if (isLoading) {
    return (
      <div className="metrics-card loading">
        <div className="loading-spinner"></div>
        <p>Loading system metrics...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="metrics-card error">
        <div className="metrics-header">
          <Server className="icon error" />
          <h2>System Metrics</h2>
        </div>
        <div className="error-message">
          ❌ Failed to fetch system metrics
          <p className="error-details">{error?.toString()}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="metrics-card">
      <div className="metrics-header">
        <Server className="icon" />
        <h2>System Metrics</h2>
      </div>
      
      <div className="metrics-content">
        <div className="metrics-overview">
          <div className="metric-item">
            <TrendingUp className="metric-icon" />
            <div>
              <span className="metric-label">Avg Response Time</span>
              <span className="metric-value">{metrics?.metrics.averageResponseTime}</span>
            </div>
          </div>
          
          <div className="metric-item">
            <AlertTriangle className="metric-icon" />
            <div>
              <span className="metric-label">Error Rate</span>
              <span className="metric-value">
                {metrics ? ((metrics.metrics.errorCount / Math.max(metrics.metrics.requestCount, 1)) * 100).toFixed(2) : 0}%
              </span>
            </div>
          </div>
        </div>

        <div className="charts-container">
          <div className="chart-section">
            <h3>Memory Usage</h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={memoryData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {memoryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value}%`} />
              </PieChart>
            </ResponsiveContainer>
            <div className="memory-details">
              <p>Used: {metrics?.system.memory.used}</p>
              <p>Total: {metrics?.system.memory.total}</p>
            </div>
          </div>

          <div className="chart-section">
            <h3>Request Statistics</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={requestData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="system-info">
          <h3>System Information</h3>
          <div className="info-grid">
            <div><strong>Platform:</strong> {metrics?.system.platform}</div>
            <div><strong>Node Version:</strong> {metrics?.system.nodeVersion}</div>
            <div><strong>PID:</strong> {metrics?.system.pid}</div>
            <div><strong>Uptime:</strong> {metrics?.server.uptime}</div>
          </div>
        </div>
      </div>
    </div>
  );
};
