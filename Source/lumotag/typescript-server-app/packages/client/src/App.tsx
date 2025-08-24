// React import not needed with JSX Transform
import { QueryClient, QueryClientProvider } from 'react-query';
import { ServerStatus } from '@/components/ServerStatus';
import { SystemMetrics } from '@/components/SystemMetrics';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Refetch every 5 seconds
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <header className="App-header">
          <h1>Server Status Dashboard</h1>
          <p>Real-time monitoring of your TypeScript server</p>
        </header>
        
        <main className="App-main">
          <div className="dashboard-grid">
            <ServerStatus />
            <SystemMetrics />
          </div>
        </main>
        
        <footer className="App-footer">
          <p>TypeScript Server Dashboard v1.0.0</p>
        </footer>
      </div>
    </QueryClientProvider>
  );
}

export default App;
