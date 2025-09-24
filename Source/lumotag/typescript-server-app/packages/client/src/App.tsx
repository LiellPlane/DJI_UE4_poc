// React import not needed with JSX Transform
import { QueryClient, QueryClientProvider } from "react-query";
import { PlayerDashboard } from "@/components/PlayerDashboard";
import { ServerMetricsComponent } from "@/components/ServerMetrics";
import { ImageActivity } from "@/components/ImageActivity";
import { TestPanel } from "@/components/TestPanel";
import "./App.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: false, // Disable global refetch, each component handles its own
      refetchOnWindowFocus: false,
      retry: 2,
      staleTime: 1000, // Consider data stale after 1 second
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <header className="App-header">
          <h1>Lumotag Server Dashboard</h1>
        </header>

        <main className="App-main">
          <div className="dashboard-grid">
            <PlayerDashboard />
            <ServerMetricsComponent />
            <ImageActivity />
          </div>
          <TestPanel />
        </main>

        <footer className="App-footer">
          <p>Game Dashboard v1.0.0</p>
        </footer>
      </div>
    </QueryClientProvider>
  );
}

export default App;
