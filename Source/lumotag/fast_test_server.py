#!/usr/bin/env python3
"""
Ultra-fast test server using FastAPI for performance comparison
"""
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Simple static gamestate - no calculations
STATIC_GAMESTATE = {
    "players": [
        {
            "tag_id": "player_001",
            "display_name": "Player 1", 
            "health": 100,
            "ammo": 30,
            "event_type": "PlayerStatus"
        },
        {
            "tag_id": "player_002", 
            "display_name": "Player 2",
            "health": 85,
            "ammo": 22,
            "event_type": "PlayerStatus"
        }
    ],
    "server_time": 0,
    "game_mode": "team_deathmatch",
    "map": "urban_warfare"
}

@app.get("/api/v1/gamestate")
async def get_gamestate():
    """Ultra-fast gamestate endpoint - no calculations, just return cached data"""
    # Only update timestamp - everything else is static for speed test
    STATIC_GAMESTATE["server_time"] = time.time()
    return JSONResponse(STATIC_GAMESTATE)

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    print("🚀 Starting FastAPI test server on http://127.0.0.1:8899")
    print("   Gamestate: http://127.0.0.1:8899/api/v1/gamestate")
    print("   Health: http://127.0.0.1:8899/health")
    print("")
    print("Compare this with your HTTPServer performance!")
    
    # Run with minimal overhead
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8899,
        log_level="error",  # Minimal logging
        access_log=False    # No access logs for max speed
    )
