"""
FastAPI application with WebSocket support for game state management.

This module provides a FastAPI server with WebSocket capabilities
for real-time game state updates between clients.
"""

import logging
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Lumotag Server",
    description="FastAPI server for real-time game state management",
    version="0.1.0",
)


class GameStateManager:
    """Manages game state and WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.game_state: Dict[str, Any] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection and add it to active connections."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"New WebSocket connection. "
            f"Total connections: {len(self.active_connections)}"
        )
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from active connections."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. "
                f"Total connections: {len(self.active_connections)}"
            )
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a personal message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast_game_state(self, game_state: Dict[str, Any]):
        """Broadcast game state to all active WebSocket connections."""
        disconnected_connections = []
        message = f"GAME_STATE:{game_state}"
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected_connections.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected_connections:
            self.disconnect(connection)
    
    def update_game_state(self, key: str, value: Any):
        """Update a specific key in the game state."""
        self.game_state[key] = value
        logger.info(f"Game state updated: {key} = {value}")
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get the current game state."""
        return self.game_state.copy()


# Global game state manager instance
game_manager = GameStateManager()


@app.get("/")
async def root():
    """Root endpoint returning basic information about the server."""
    return {
        "message": "Lumotag Game Server",
        "version": "0.1.0",
        "active_connections": len(game_manager.active_connections),
        "game_state": game_manager.get_game_state()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "active_connections": len(game_manager.active_connections)
    }


@app.get("/game-state")
async def get_game_state():
    """Get the current game state."""
    return {
        "game_state": game_manager.get_game_state(),
        "active_connections": len(game_manager.active_connections)
    }


@app.post("/game-state")
async def update_game_state(update: Dict[str, Any]):
    """Update game state and broadcast to all connected clients."""
    if not update:
        raise HTTPException(
            status_code=400, 
            detail="Update data is required"
        )
    
    # Update game state
    for key, value in update.items():
        game_manager.update_game_state(key, value)
    
    # Broadcast updated state to all clients
    await game_manager.broadcast_game_state(game_manager.get_game_state())
    
    return {
        "status": "Game state updated", 
        "active_connections": len(game_manager.active_connections),
        "updated_fields": list(update.keys())
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time game state communication."""
    await game_manager.connect(websocket)
    
    try:
        # Send current game state to newly connected client
        current_state = game_manager.get_game_state()
        if current_state:
            await game_manager.send_personal_message(
                f"GAME_STATE:{current_state}", 
                websocket
            )
        
        # Send welcome message
        await game_manager.send_personal_message(
            "CONNECTED:Welcome to Lumotag Game Server", 
            websocket
        )
        
        while True:
            # Wait for message from client
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            # Parse game state updates
            if data.startswith("UPDATE:"):
                try:
                    # Extract update data (expecting JSON-like format)
                    update_data = data[7:]  # Remove "UPDATE:" prefix
                    # Here you would parse the update data as needed
                    # For now, we'll just acknowledge receipt
                    await game_manager.send_personal_message(
                        f"ACK:{update_data}", 
                        websocket
                    )
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    await game_manager.send_personal_message(
                        f"ERROR:Invalid update format", 
                        websocket
                    )
            else:
                # Echo other messages
                await game_manager.send_personal_message(
                    f"ECHO:{data}", 
                    websocket
                )
            
    except WebSocketDisconnect:
        game_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        game_manager.disconnect(websocket)


def main():
    """Main function to run the FastAPI server."""
    logger.info("Starting Lumotag Game Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
