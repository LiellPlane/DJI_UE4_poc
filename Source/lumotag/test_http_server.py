#!/usr/bin/env python3
"""
Standalone HTTP test server for HTTPComms smoke testing

This server implements the same endpoints as the game server would,
providing realistic responses for testing HTTPComms functionality.

Run with: python test_http_server.py
Then point your HTTPComms to: http://localhost:8080/api/v1/
"""
import math
import json
import time
import base64
import threading
import test_http_save_testimages
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from lumotag_events import GameStatus, PlayerStatus
from typing import Dict
import sys


class GameTestServer(BaseHTTPRequestHandler):
    # Class variables to track server state
    images_received = []
    events_received = []
    stored_images: Dict[str, bytes] = {}  # Dictionary to store JPG images by image_id
    players_data = {
        "testself": PlayerStatus(
            health=100,
            ammo=30,
            tag_id="testself",
            display_name="tinytim"
        ),
        "player_002": PlayerStatus(
            health=85,
            ammo=22,
            tag_id="player_002", 
            display_name="mongo"
        ),
        "player_003": PlayerStatus(
            health=95,
            ammo=18,
            tag_id="player_003",
            display_name="dildort"
        )
    }
    
    def log_message(self, format, *args):
        """Custom logging with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {format % args}")
    
    def do_GET(self):
        """Handle GET requests - mainly gamestate"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if self.path == '/api/v1/gamestate':
            self._handle_gamestate_request()
        elif self.path == '/health':
            self._handle_health_check()
        elif self.path == '/stats':
            self._handle_stats_request()
        else:
            print(f"[{timestamp}] ❌ GET 404: {self.path}")
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())
    
    def do_POST(self):
        """Handle POST requests - images and events"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                print(f"[{timestamp}] ❌ POST: Empty request body")
                self.send_response(400)
                self.end_headers()
                return
                
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Log request details
            user_id = self.headers.get('X-User-ID', 'unknown')
            print(f"[{timestamp}] 📨 POST {self.path} from {user_id}")
            
            if self.path == '/api/v1/images/upload':
                self._handle_image_upload(data)
            elif self.path == '/api/v1/events':
                self._handle_event_submission(data)
            else:
                print(f"[{timestamp}] ❌ POST 404: {self.path}")
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode())
                
        except json.JSONDecodeError as e:
            print(f"[{timestamp}] ❌ Invalid JSON: {e}")
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            
        except Exception as e:
            print(f"[{timestamp}] ❌ Server error: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
    
    def _handle_gamestate_request(self):
        """Handle gamestate polling requests"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Simulate some dynamic game state changes
        current_time = time.time()
        for tag_id, player in self.players_data.items():
            # Simulate health/ammo changes over time
            if tag_id == "testself":
                # Make testself health jiggle more noticeably for smoke testing
                
                base_health = 75
                # Use sine wave for smooth jiggling + some randomness
                sine_variation = int(20 * math.sin(current_time * 2))  # +/- 20 health variation
                random_jitter = int(5 * math.sin(current_time * 7))   # Additional jitter
                new_health = max(20, min(100, base_health + sine_variation + random_jitter))
            else:
                base_health = 85 if tag_id == "player_002" else 95
                health_variation = int(10 * (0.5 - (current_time % 10) / 20))  # +/- 5 health variation
                new_health = max(10, min(100, base_health + health_variation))
            
            # Simulate ammo consumption
            ammo_consumed = int(current_time / 5) % 5  # Consume ammo over time
            base_ammo = 30 if tag_id == "testself" else (22 if tag_id == "player_002" else 18)
            new_ammo = max(0, base_ammo - ammo_consumed)
            
            # Update player status (create new PlayerStatus since Pydantic models are immutable)
            self.players_data[tag_id] = PlayerStatus(
                health=new_health,
                ammo=new_ammo,
                tag_id=player.tag_id,
                display_name=player.display_name
            )
        
        # Create proper GameStatus response using Pydantic model
        game_update = GameStatus(players=self.players_data.copy())
        gamestate_response = game_update.model_dump()
        
        # Minimal logging for performance - only log occasionally
        if not hasattr(self, '_gamestate_counter'):
            self._gamestate_counter = 0
        self._gamestate_counter += 1
            
        # Only log every 10th request to reduce console spam
        if self._gamestate_counter % 10 == 0:
            print(f"[{timestamp}] 🎮 Gamestate request #{self._gamestate_counter} - returning {len(self.players_data)} players")
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(gamestate_response).encode())
    
    def _handle_image_upload(self, data):
        """Handle image upload requests"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Get user info from headers
        user_id = self.headers.get('X-User-ID', 'unknown')
        
        # Validate required fields (now based on UploadRequest Pydantic model)
        required_fields = ['image_id', 'image_data']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"[{timestamp}] ❌ Image upload missing fields: {missing_fields}")
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Missing fields: {missing_fields}"}).encode())
            return
        
        # Validate and decode image data
        try:
            image_data = base64.b64decode(data['image_data'])
            image_size = len(image_data)
            
            # Basic JPEG validation (check for JPEG magic bytes)
            if not image_data.startswith(b'\xff\xd8\xff'):
                print(f"[{timestamp}] ❌ Invalid JPEG format for image {data['image_id']}")
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JPEG format"}).encode())
                return
            
        except Exception as e:
            print(f"[{timestamp}] ❌ Image decode error: {e}")
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid base64 image data"}).encode())
            return
        
        # Extract image ID from the data (like HTTPComms does)
        image_id = data['image_id']
        
        # Store the JPG encoded image data in class dictionary
        GameTestServer.stored_images[image_id] = image_data
        
        # Store image info
        image_info = {
            'image_id': image_id,
            'user_id': user_id,  # From headers
            'timestamp': data.get('timestamp', time.time()),  # Use current time if not provided
            'size_bytes': image_size,
            'received_at': time.time()
        }
        
        # Display image using OpenCV for real-time monitoring
        try:
            import cv2
            import numpy as np
            
            # Most efficient: decode JPEG directly from bytes buffer
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            test_http_save_testimages.test_saver.save_image_pair(img, image_id)
            if img is not None:


                # Resize image to height 800px while maintaining aspect ratio
                height, width = img.shape[:2]
                if height != 800:
                    aspect_ratio = width / height
                    new_width = int(800 * aspect_ratio)
                    img = cv2.resize(img, (new_width, 800), interpolation=cv2.INTER_LINEAR)
                
                # Display image without any blocking - cv2.waitKey(1) is non-blocking
                # cv2.imshow(f'Received Images - {user_id}', img)
                cv2.waitKey(1)  # Non-blocking, just processes window events
        except ImportError:
            # OpenCV not available, skip display
            pass
        except Exception as e:
            # Don't let display errors crash the server
            print(f"[{timestamp}] ⚠️  Image display error: {e}")
        
        GameTestServer.images_received.append(image_info)
        
        print(f"[{timestamp}] 📸 Image uploaded successfully:")
        print(f"    🔍 ID: {image_id}")
        print(f"    👤 User: {user_id}")
        print(f"    📏 Size: {image_size:,} bytes")
        print(f"    🕐 Client timestamp: {datetime.fromtimestamp(data.get('timestamp', time.time())).strftime('%H:%M:%S')}")
        
        # Simulate some processing time
        time.sleep(0.01)  # 10ms processing delay
        
        response = {
            "status": "success",
            "image_id": image_id,
            "processed_at": time.time(),
            "message": "Image processed successfully"
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_event_submission(self, data):
        """Handle event submission requests"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Get user info from headers
        user_id = self.headers.get('X-User-ID', 'unknown')
        
        # Validate event data structure
        if not isinstance(data, dict):
            print(f"[{timestamp}] ❌ Event data must be a dictionary")
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Event data must be a dictionary"}).encode())
            return
        
        # Extract event type
        event_type = data.get('event_type', 'unknown')
        
        # Store event info
        event_info = {
            'event_type': event_type,
            'user_id': user_id,
            'data': data.copy(),
            'received_at': time.time()
        }
        
        GameTestServer.events_received.append(event_info)
        
        print(f"[{timestamp}] 📨 Event received successfully:")
        print(f"    🏷️  Type: {event_type}")
        print(f"    👤 User: {user_id}")
        print(f"    📦 Data: {data}")
        
        response = {
            "status": "success",
            "event_type": event_type,
            "processed_at": time.time(),
            "message": "Event processed successfully"
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_health_check(self):
        """Handle health check requests"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] ❤️  Health check request")
        
        health_info = {
            "status": "healthy",
            "uptime_seconds": time.time() - server_start_time,
            "images_received": len(GameTestServer.images_received),
            "events_received": len(GameTestServer.events_received),
            "active_players": len(GameTestServer.players_data)
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health_info).encode())
    
    def _handle_stats_request(self):
        """Handle stats requests"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] 📊 Stats request")
        
        stats = {
            "server_info": {
                "uptime_seconds": time.time() - server_start_time,
                "start_time": datetime.fromtimestamp(server_start_time).isoformat()
            },
            "activity": {
                "total_images": len(GameTestServer.images_received),
                "total_events": len(GameTestServer.events_received),
                "recent_images": GameTestServer.images_received[-5:],  # Last 5 images
                "recent_events": GameTestServer.events_received[-5:]   # Last 5 events
            },
            "game_state": {
                "active_players": len(GameTestServer.players_data),
                "players": {tag_id: player.model_dump() for tag_id, player in GameTestServer.players_data.items()}
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats, indent=2).encode())


def run_server(host='0.0.0.0', port=8080):
    """Run the test server"""
    global server_start_time
    server_start_time = time.time()
    
    print("🚀 Starting HTTPComms Test Server...")
    print(f"📍 Server URL: http://{host}:{port}")
    print(f"🎮 Gamestate endpoint: http://{host}:{port}/api/v1/gamestate")
    print(f"📸 Images endpoint: http://{host}:{port}/api/v1/images/upload")
    print(f"📨 Events endpoint: http://{host}:{port}/api/v1/events")
    print(f"❤️  Health check: http://{host}:{port}/health")
    print(f"📊 Stats: http://{host}:{port}/stats")
    print()
    print("💡 Configure your HTTPComms with these URLs:")
    print(f"   images_url='http://{host}:{port}/api/v1/images/upload'")
    print(f"   events_url='http://{host}:{port}/api/v1/events'")
    print(f"   gamestate_url='http://{host}:{port}/api/v1/gamestate'")
    print()
    print("🔄 Server is running... Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        server = HTTPServer((host, port), GameTestServer)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
        server.shutdown()
        
        # Print final stats
        print("\n📊 Final Statistics:")
        print(f"   📸 Total images received: {len(GameTestServer.images_received)}")
        print(f"   📨 Total events received: {len(GameTestServer.events_received)}")
        print(f"   ⏱️  Server uptime: {time.time() - server_start_time:.1f} seconds")
        print("✨ Goodbye!")


if __name__ == "__main__":
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    
    run_server(host, port)
