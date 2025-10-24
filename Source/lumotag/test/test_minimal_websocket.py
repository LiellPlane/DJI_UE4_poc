#!/usr/bin/env python3
"""
Minimal WebSocket Server for Image Testing

This is a bare-bones websocket server that only:
1. Accepts websocket connections
2. Receives JSON messages
3. Logs image uploads with basic info
4. Does NO image display, NO broadcasting, NO complex validation

Use this to isolate whether the issue is in the complex fake server logic.
"""
import asyncio
import websockets
import json
import base64
import time
import socket
from typing import Dict, Any


class MinimalImageServer:
    def __init__(self, host='0.0.0.0', port=None):
        self.host = host
        self.port = port or self._find_free_port()
        self.server = None
        self.is_running = False
        self.stats = {
            'connections': 0,
            'messages_received': 0,
            'image_uploads': 0,
            'errors': 0
        }
        
    def _find_free_port(self, start_port=8765):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No free ports found")
    
    async def handle_client(self, websocket, path=None):
        """Handle incoming WebSocket connections - minimal processing"""
        self.stats['connections'] += 1
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"🔗 MinimalWS: Client connected from {client_addr}")
        
        try:
            async for message in websocket:
                self.stats['messages_received'] += 1
                await self._process_message(message, client_addr)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 MinimalWS: Client {client_addr} disconnected")
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ MinimalWS: Error handling client {client_addr}: {e}")
    
    async def _process_message(self, message: str, client_addr: str):
        """Process incoming message - minimal validation"""
        try:
            # Parse JSON
            data = json.loads(message)
            
            # Check if it looks like an image upload
            if data.get('type') == 'image_upload' or 'image_data' in data:
                self.stats['image_uploads'] += 1
                
                # Extract basic info without validation
                image_id = data.get('image_id', 'unknown')
                timestamp = data.get('timestamp', 0)
                image_data = data.get('image_data', '')
                
                # Get image size if possible
                img_size = 0
                if image_data:
                    try:
                        img_bytes = base64.b64decode(image_data)
                        img_size = len(img_bytes)
                    except Exception:
                        img_size = -1  # Invalid base64
                
                # Simple logging - no display, no complex processing
                current_time = time.strftime('%H:%M:%S')
                print(f"📷 [{current_time}] Image from {client_addr}: "
                      f"id={image_id}, size={img_size}bytes, ts={timestamp}")
                
                # Show stats every 10 images
                if self.stats['image_uploads'] % 10 == 0:
                    print(f"📊 MinimalWS Stats: {self.stats}")
            else:
                # Non-image message
                msg_type = data.get('type', data.get('event_type', 'unknown'))
                print(f"📨 MinimalWS: Non-image message from {client_addr}: {msg_type}")
                
        except json.JSONDecodeError:
            self.stats['errors'] += 1
            print(f"❌ MinimalWS: Invalid JSON from {client_addr}")
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ MinimalWS: Error processing message from {client_addr}: {e}")
    
    async def start_async(self):
        """Start the WebSocket server asynchronously"""
        print(f"🚀 MinimalWS: Starting server on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port
        )
        self.is_running = True
        
        print(f"✅ MinimalWS: Server running on ws://{self.host}:{self.port}")
        print(f"📝 MinimalWS: This server only logs images - no display, no broadcasting")
        
        # Keep server running
        await self.server.wait_closed()
    
    def get_url(self):
        """Get the WebSocket URL for this server"""
        return f"ws://{self.host}:{self.port}"
    
    def get_stats(self):
        """Get server statistics"""
        return self.stats.copy()


async def main():
    """Run the minimal server"""
    server = MinimalImageServer()
    
    try:
        await server.start_async()
    except KeyboardInterrupt:
        print("\n🛑 MinimalWS: Shutting down...")
        if server.server:
            server.server.close()
            await server.server.wait_closed()
        print(f"📊 Final stats: {server.get_stats()}")


if __name__ == "__main__":
    print("🧪 MinimalWS: Starting minimal image testing server...")
    print("🎯 Purpose: Test if images are received without complex processing")
    print("⚡ Features: JSON parsing, image detection, basic logging only")
    print("🚫 No features: OpenCV display, broadcasting, complex validation")
    print()
    
    asyncio.run(main())
