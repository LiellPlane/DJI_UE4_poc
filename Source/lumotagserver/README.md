# Lumotag Game Server

A FastAPI server with WebSocket support for real-time game state management.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **WebSocket Support**: Real-time bidirectional communication
- **Game State Management**: Centralised game state with automatic broadcasting
- **Connection Management**: Automatic handling of WebSocket connections
- **Health Monitoring**: Server status and connection tracking
- **RESTful API**: HTTP endpoints for game state operations

## Installation

1. Ensure you have Python 3.11+ installed
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

## Running the Server

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## API Endpoints

### HTTP Endpoints

- `GET /` - Server information and current game state
- `GET /health` - Health check endpoint
- `GET /game-state` - Get current game state
- `POST /game-state` - Update game state and broadcast to all clients

### WebSocket Endpoint

- `ws://localhost:8000/ws` - WebSocket connection for real-time updates

## Game State Management

The server maintains a centralised game state that can be updated via HTTP POST requests or WebSocket messages. When the game state is updated, it's automatically broadcasted to all connected WebSocket clients.

## Usage Examples

### Connecting a Game Client

```python
import asyncio
import websockets
import json

async def game_client():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Listen for game state updates
        async for message in websocket:
            if message.startswith("GAME_STATE:"):
                game_state = message[11:]  # Remove "GAME_STATE:" prefix
                print(f"Game state updated: {game_state}")
            elif message.startswith("CONNECTED:"):
                print("Connected to game server")
            elif message.startswith("ACK:"):
                print("Update acknowledged")

asyncio.run(game_client())
```

### Updating Game State via HTTP

```bash
curl -X POST "http://localhost:8000/game-state" \
     -H "Content-Type: application/json" \
     -d '{"player_position": {"x": 100, "y": 200}, "score": 1500}'
```

### Sending Updates via WebSocket

```python
import asyncio
import websockets

async def send_update():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Send a game state update
        await websocket.send("UPDATE:player_moved:x=150,y=250")
        
        # Wait for acknowledgement
        response = await websocket.recv()
        print(f"Server response: {response}")

asyncio.run(send_update())
```

### Getting Current Game State

```bash
curl "http://localhost:8000/game-state"
```

## Message Protocol

### WebSocket Messages

**From Client to Server:**
- `UPDATE:<data>` - Send game state update
- Any other message - Echoed back by server

**From Server to Client:**
- `GAME_STATE:<json>` - Current game state
- `CONNECTED:<message>` - Connection confirmation
- `ACK:<data>` - Update acknowledgement
- `ECHO:<data>` - Echo of non-update messages
- `ERROR:<message>` - Error messages

### HTTP API

**GET /game-state**
```json
{
  "game_state": {...},
  "active_connections": 2
}
```

**POST /game-state**
```json
{
  "player_position": {"x": 100, "y": 200},
  "score": 1500
}
```

Response:
```json
{
  "status": "Game state updated",
  "active_connections": 2,
  "updated_fields": ["player_position", "score"]
}
```

## Project Structure

```
lumotagserver/
├── main.py          # Main FastAPI application
├── pyproject.toml   # Project configuration and dependencies
├── README.md        # This file
└── uv.lock         # Dependency lock file
```

## Development

The server runs with auto-reload enabled in development mode, so changes to the code will automatically restart the server.

## Logging

The server uses Python's built-in logging module with INFO level by default. All WebSocket connections, disconnections, and game state updates are logged.

## Error Handling

- WebSocket connections are automatically cleaned up on disconnect
- Game state broadcast errors are handled gracefully
- Invalid updates return appropriate error messages
- All endpoints include proper error handling and HTTP status codes
