#!/usr/bin/env python3
"""
Fake WebSocket Server for Smoke Testing

Auto-starts a WebSocket server on import for testing comms.py
Consumes and validates incoming image upload messages, then drops them.
No linting required - this is a smoke testing module.
"""
import traceback
import asyncio
import websockets
import json
import base64
import threading
import time
import socket
from typing import Dict, Any

class FakeWebSocketServer:
    def __init__(self, host='127.0.0.1', port=None):
        self.host = host
        self.port = port or self._find_free_port()
        self.server = None
        self.loop = None
        self.server_thread = None
        self.is_running = False
        self.stats = {
            'messages_received': 0,
            'valid_uploads': 0,
            'invalid_messages': 0,
            'connections': 0
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
        """Handle incoming WebSocket connections"""
        self.stats['connections'] += 1
        try:
            client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except:
            client_addr = "unknown"
        print(f"🔗 FakeWS: Client connected from {client_addr}")
        
        try:
            async for message in websocket:
                self.stats['messages_received'] += 1
                await self._process_message(message, client_addr)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 FakeWS: Client {client_addr} disconnected")
        except Exception as e:
            print(f"❌ FakeWS: Error handling client {client_addr}: {e}")
            
            traceback.print_exc()
    
    async def _process_message(self, message: str, client_addr: str):
        """Process and validate incoming messages"""
        try:
            # Parse JSON
            data = json.loads(message)
            
            # Validate message structure
            if not self._validate_upload_message(data):
                self.stats['invalid_messages'] += 1
                print(f"❌ FakeWS: Invalid message from {client_addr}")
                return
            
            # Message is valid - log it and drop it
            self.stats['valid_uploads'] += 1
            image_id = data.get('image_id', 'unknown')
            timestamp = data.get('timestamp', 0)
            image_data = data.get('image_data', '')
            
            # Decode image data to get size (but don't store it)
            try:
                img_bytes = base64.b64decode(image_data)
                img_size = len(img_bytes)
            except:
                img_size = -1
                
            print(f"✅ FakeWS: Valid upload from {client_addr}")
            print(f"   🌐 server: {self.host}:{self.port}")
            print(f"   📷 image_id: {image_id}")
            print(f"   ⏰ timestamp: {timestamp}")
            print(f"   📦 image_size: {img_size} bytes")
            print(f"   📊 Stats: {self.stats['valid_uploads']} valid, {self.stats['invalid_messages']} invalid")
            
        except json.JSONDecodeError as e:
            self.stats['invalid_messages'] += 1
            print(f"❌ FakeWS: JSON decode error from {client_addr}: {e}")
        except Exception as e:
            self.stats['invalid_messages'] += 1
            print(f"❌ FakeWS: Unexpected error processing message from {client_addr}: {e}")
    
    def _validate_upload_message(self, data: Dict[str, Any]) -> bool:
        """Validate the structure of an upload message"""
        if not isinstance(data, dict):
            return False
            
        # Check required fields
        required_fields = ['type', 'image_id', 'timestamp', 'image_data']
        for field in required_fields:
            if field not in data:
                print(f"❌ FakeWS: Missing required field: {field}")
                return False
        
        # Check message type
        if data.get('type') != 'image_upload':
            print(f"❌ FakeWS: Invalid message type: {data.get('type')}")
            return False
            
        # Check field types
        if not isinstance(data.get('image_id'), str):
            print(f"❌ FakeWS: image_id must be string")
            return False
            
        if not isinstance(data.get('timestamp'), (int, float)):
            print(f"❌ FakeWS: timestamp must be number")
            return False
            
        if not isinstance(data.get('image_data'), str):
            print(f"❌ FakeWS: image_data must be string")
            return False
        
        # Validate base64 encoding
        try:
            base64.b64decode(data.get('image_data'))
        except Exception:
            print(f"❌ FakeWS: image_data is not valid base64")
            return False
            
        return True
    
    def start(self):
        """Start the WebSocket server in a background thread"""
        if self.is_running:
            return
            
        print(f"🔄 FakeWS: Starting server on {self.host}:{self.port}...")
        
        def run_server():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
                async def server_main():
                    try:
                        self.server = await websockets.serve(
                            self.handle_client, 
                            self.host, 
                            self.port
                        )
                        self.is_running = True
                        print(f"🚀 FakeWS: Server started on {self.host}:{self.port}")
                        print(f"🔗 FakeWS: Use URL: ws://{self.host}:{self.port}")
                        await self.server.wait_closed()
                    except Exception as e:
                        print(f"❌ FakeWS: Server main error: {e}")
                        traceback.print_exc()
                        
                self.loop.run_until_complete(server_main())
            except Exception as e:
                print(f"❌ FakeWS: Server thread error: {e}")
                traceback.print_exc()
            finally:
                self.is_running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start with more generous timeout
        print("🔄 FakeWS: Waiting for server to start...")
        for i in range(100):  # Wait up to 10 seconds
            if self.is_running:
                print(f"✅ FakeWS: Server started after {(i+1)*0.1:.1f}s")
                break
            time.sleep(0.1)
        
        if not self.is_running:
            print("❌ FakeWS: Server failed to start within timeout")
            # Don't raise exception - just warn and continue
            print("⚠️ FakeWS: Continuing without WebSocket server (connections will fail)")
            return False
        
        return True
    
    def stop(self):
        """Stop the WebSocket server"""
        if self.server and self.is_running:
            if self.loop:
                self.loop.call_soon_threadsafe(self.server.close)
                self.is_running = False
                print(f"🛑 FakeWS: Server stopped")
    
    def get_url(self):
        """Get the WebSocket URL for this server"""
        return f"ws://{self.host}:{self.port}"
    
    def get_stats(self):
        """Get server statistics"""
        return self.stats.copy()

# Global server instance - auto-start on import
_fake_server = None

def get_fake_server():
    """Get the global fake server instance, starting it if needed"""
    global _fake_server
    if _fake_server is None:
        _fake_server = FakeWebSocketServer()
        _fake_server.start()
    return _fake_server

def get_fake_websocket_url():
    """Get the URL of the fake WebSocket server"""
    server = get_fake_server()
    return server.get_url()

def get_server_stats():
    """Get statistics from the fake server"""
    server = get_fake_server()
    return server.get_stats()

# Auto-start server when module is imported
print("🔄 FakeWS: Auto-starting fake WebSocket server...")
try:
    _auto_server = get_fake_server()
    if _auto_server.is_running:
        print(f"✅ FakeWS: Ready! Use URL: {_auto_server.get_url()}")
        FAKE_WEBSOCKET_URL = _auto_server.get_url()
    else:
        print("⚠️ FakeWS: Server failed to start, using fallback URL")
        FAKE_WEBSOCKET_URL = "ws://127.0.0.1:8765"  # Fallback URL
    fake_server = _auto_server
except Exception as e:
    print(f"❌ FakeWS: Failed to start server: {e}")
    print("⚠️ FakeWS: Using fallback URL (connections will fail until server starts)")
    FAKE_WEBSOCKET_URL = "ws://127.0.0.1:8765"  # Fallback URL
    fake_server = None

if __name__ == "__main__":
    print("🧪 FakeWS: Running in test mode...")
    
    # Keep server running for manual testing
    try:
        while True:
            time.sleep(1)
            stats = get_server_stats()
            if stats['messages_received'] > 0:
                print(f"📊 FakeWS: Stats: {stats}")
    except KeyboardInterrupt:
        print("\n🛑 FakeWS: Shutting down...")
        _fake_server.stop()
