# TypeScript Server with React Dashboard

A modern full-stack TypeScript application featuring an Express.js server with a React frontend for real-time server status monitoring.

## Features

- **Modern TypeScript Stack**: Full TypeScript setup with strict type checking
- **Express.js Server**: Secure, production-ready API server with middleware
- **React Dashboard**: Real-time monitoring dashboard with beautiful UI
- **Health Monitoring**: Comprehensive health checks and system metrics
- **Real-time Updates**: Auto-refreshing status with React Query
- **Modern Tooling**: Vite, ESLint, Prettier, pnpm workspaces
- **Security**: Helmet, CORS, input validation with Zod
- **Logging**: Structured logging with Winston
- **Charts & Visualizations**: Interactive charts with Recharts
- **Responsive Design**: Mobile-friendly dashboard

## 📁 Project Structure

```
typescript-server-app/
├── packages/
│   ├── server/                 # Express.js TypeScript server
│   │   ├── src/
│   │   │   ├── routes/         # API routes (health, status)
│   │   │   ├── middlewares/    # Custom middleware
│   │   │   ├── utils/          # Utilities (logger)
│   │   │   ├── types/          # TypeScript types
│   │   │   └── app.ts          # Main server entry
│   │   ├── dist/               # Compiled output
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── client/                 # React frontend
│       ├── src/
│       │   ├── components/     # React components
│       │   ├── services/       # API services
│       │   ├── types/          # TypeScript types
│       │   └── App.tsx         # Main React app
│       ├── package.json
│       └── vite.config.ts
├── package.json                # Root monorepo config
├── pnpm-workspace.yaml        # pnpm workspace config
└── README.md
```

## Prerequisites

- **Node.js** >= 18.0.0
- **pnpm** >= 8.0.0 (recommended) or npm/yarn

Install pnpm globally:
```bash
npm install -g pnpm
```

## Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd typescript-server-app
   ```

2. **Install dependencies:**
   ```bash
   pnpm install
   ```

3. **Start development servers:**
   ```bash
   # Start both server and client in parallel
   pnpm dev
   ```
   
   Or start them individually:
   ```bash
   # Terminal 1: Start backend server (port 3000)
   pnpm --filter server dev
   
   # Terminal 2: Start frontend client (port 3001)
   pnpm --filter client dev
   ```

4. **Open your browser:**
   - Frontend Dashboard: http://localhost:3001
   - Backend API: http://localhost:3000
   - Health Check: http://localhost:3000/api/health
   - System Status: http://localhost:3000/api/status

## API Endpoints

### Health Endpoints
- `GET /api/health` - Basic health check
- `GET /api/health?detailed=true` - Detailed health information
- `GET /api/health/ready` - Readiness probe
- `GET /api/health/live` - Liveness probe

### Status Endpoints
- `GET /api/status` - Comprehensive system metrics
- `GET /` - API information

### Example Response
```json
{
  "status": "UP",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "uptime": 3600,
  "memory": {
    "rss": 45678592,
    "heapTotal": 20971520,
    "heapUsed": 15728640,
    "external": 1048576
  },
  "environment": "development",
  "version": "1.0.0"
}
```

## Build & Deployment

### Development
```bash
# Start development with hot reload
pnpm dev

# Run tests
pnpm test

# Lint code
pnpm lint

# Format code
pnpm format
```

### Production Build
```bash
# Build both server and client
pnpm build

# Start production server
pnpm start
```

### Individual Package Commands
```bash
# Server commands
pnpm --filter server build
pnpm --filter server start
pnpm --filter server dev

# Client commands
pnpm --filter client build
pnpm --filter client dev
pnpm --filter client preview
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Server Configuration
PORT=3000
NODE_ENV=development
FRONTEND_URL=http://localhost:3001

# Logging
LOG_LEVEL=info

# Add more as needed...
```

### Server Configuration
- **Port**: Configurable via `PORT` environment variable (default: 3000)
- **CORS**: Configured to allow requests from frontend URL
- **Security**: Helmet middleware with CSP policies
- **Logging**: Winston with file rotation and console output

### Client Configuration
- **API URL**: Automatically proxies `/api` requests to backend
- **Auto-refresh**: Dashboard updates every 5-10 seconds
- **Responsive**: Mobile-friendly design

## Dashboard Features

The React dashboard provides:

### Server Status Card
- Server UP/DOWN status
- Uptime tracking
- Memory usage
- Environment info
- Last update timestamp

### System Metrics Card
- Memory usage pie chart
- Request statistics bar chart
- Average response time
- Error rate monitoring
- System information (platform, Node version, PID)

## Security Features

- **Helmet**: Security headers and CSP policies
- **CORS**: Configurable cross-origin resource sharing
- **Input Validation**: Zod schemas for request validation
- **Error Handling**: Centralized error middleware
- **Rate Limiting**: Ready for implementation
- **Environment Variables**: Sensitive data protection

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Run server tests
pnpm --filter server test

# Run client tests  
pnpm --filter client test

# Watch mode
pnpm --filter server test:watch
```

## Development Guidelines

### Code Style
- **ESLint**: Configured with TypeScript rules
- **Prettier**: Consistent code formatting
- **TypeScript**: Strict mode enabled
- **Imports**: Path aliases configured (`@/`)

### Git Hooks (Optional)
Add pre-commit hooks for:
- Linting
- Type checking
- Tests
- Formatting

## Production Deployment

### Docker (Recommended)
```dockerfile
# Example Dockerfile structure
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install --only=production
COPY dist/ ./dist/
EXPOSE 3000
CMD ["node", "dist/app.js"]
```

### Process Management
Use PM2 for production:
```bash
npm install -g pm2
pm2 start dist/app.js --name "typescript-server"
```

### Environment Setup
- Set `NODE_ENV=production`
- Configure proper logging levels
- Set up monitoring and alerting
- Configure reverse proxy (nginx)
- Enable HTTPS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run linting and tests
6. Submit a pull request

## 📜 Scripts Reference

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start development servers |
| `pnpm build` | Build for production |
| `pnpm start` | Start production server |
| `pnpm test` | Run tests |
| `pnpm lint` | Lint code |
| `pnpm format` | Format code |
| `pnpm clean` | Clean build artifacts |

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill process on port 3000
   lsof -ti:3000 | xargs kill -9
   ```

2. **pnpm not found**
   ```bash
   npm install -g pnpm
   ```

3. **TypeScript errors**
   ```bash
   # Clean and reinstall
   pnpm clean
   rm -rf node_modules
   pnpm install
   ```

4. **API connection issues**
   - Check if backend server is running on port 3000
   - Verify CORS configuration
   - Check network proxy settings

## 📚 Tech Stack

### Backend
- **Express.js** - Web framework
- **TypeScript** - Type safety
- **Winston** - Logging
- **Helmet** - Security
- **Cors** - Cross-origin requests
- **Zod** - Runtime validation
- **Morgan** - HTTP logging

### Frontend
- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **React Query** - Data fetching
- **Recharts** - Charts and graphs
- **Lucide React** - Icons
- **Axios** - HTTP client

### Development
- **pnpm** - Package manager
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **Jest** - Testing framework
- **ts-node-dev** - Development server

## 📄 License

This project is licensed under the MIT License.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with details
4. Include error logs and environment info

---

**Happy coding!**
