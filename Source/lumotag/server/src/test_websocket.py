import json
import asyncio
import websockets


async def test_websocket_connection():
    # Connect to the WebSocket server
    uri = "ws://127.0.0.1:8080"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Heartbeat message
            heartbeat_msg = {
                "type": "heartbeat",
                "data": {
                    "id": "player123",
                    "timestamp": 1234567890,
                    "cached_avatars": ["avatar1.png", "avatar2.png"]
                }
            }

            # Player update message
            update_msg = {
                "type": "update", 
                "data": {
                    "id": "player123",
                    "timestamp": 1234567890,
                    "cached_avatars": ["avatar1.png"]
                }
            }

            # Player state message
            state_msg = {
                "type": "state",
                "data": {
                    "id": "player123",
                    "name": "TestPlayer",
                    "healthpoints": 100,
                    # Example binary data
                    "avatar_image": [0x00, 0x01, 0x02, 0x03]
                }
            }

            # Send heartbeat message
            print("Sending heartbeat message...")
            await websocket.send(json.dumps(heartbeat_msg))
            
            # Wait a bit
            await asyncio.sleep(1)
            
            # Send update message
            print("Sending update message...")
            await websocket.send(json.dumps(update_msg))
            
            # Wait a bit
            await asyncio.sleep(1)
            
            # Send state message
            print("Sending state message...")
            await websocket.send(json.dumps(state_msg))
            
            # Listen for any responses
            print("Listening for responses...")
            try:
                async for message in websocket:
                    print(f"Received: {message}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server")
                
    except Exception as e:
        print(f"Error connecting to WebSocket server: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket_connection())