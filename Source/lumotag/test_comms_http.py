#!/usr/bin/env python3
"""
Simple tests for HTTPImageComms - much cleaner than the complex WebSocket tests in comms.py

Run with: python test_comms_http.py
"""

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import base64
import numpy as np
from lumotag_events import PlayerStatus
from my_collections import SharedMem_ImgTicket
from comms_http import HTTPComms

print("🧪 Starting HTTPImageComms Tests...")

# Test HTTP Server
class TestHTTPHandler(BaseHTTPRequestHandler):
    # Class variables to track received requests
    images_received = []
    events_received = []
    
    def log_message(self, format, *args):
        # Suppress default HTTP server logging
        pass
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/api/v1/images/upload':
                self._handle_image_upload(data)
            elif self.path == '/api/v1/events':
                self._handle_events(data)
            else:
                self.send_response(404)
                self.end_headers()
                
        except Exception as e:
            print(f"❌ Server error: {e}")
            self.send_response(500)
            self.end_headers()
    
    def do_GET(self):
        try:
            if self.path == '/api/v1/gamestate':
                self._handle_gamestate_get()
            else:
                self.send_response(404)
                self.end_headers()
                
        except Exception as e:
            print(f"❌ Server error: {e}")
            self.send_response(500)
            self.end_headers()
    
    def _handle_image_upload(self, data):
        # Validate required fields for images
        required_fields = ['user_id', 'device_name', 'image_id', 'timestamp', 'image_data']
        
        for field in required_fields:
            if field not in data:
                print(f"❌ Image upload missing field: {field}")
                self.send_response(400)  # Bad Request
                self.end_headers()
                return
        
        # Validate image_data is valid base64
        try:
            image_bytes = base64.b64decode(data['image_data'])
            if len(image_bytes) < 100:  # Minimum reasonable image size
                print(f"❌ Image too small: {len(image_bytes)} bytes")
                self.send_response(400)
                self.end_headers()
                return
        except Exception:
            print("❌ Invalid base64 image data")
            self.send_response(400)
            self.end_headers()
            return
        
        # Success - store the upload
        TestHTTPHandler.images_received.append({
            'user_id': data['user_id'],
            'device_name': data['device_name'], 
            'image_id': data['image_id'],
            'size': len(image_bytes)
        })
        
        print(f"✅ Image received: {data['image_id']} ({len(image_bytes)} bytes) from {data['user_id']}")
        self.send_response(200)
        self.end_headers()
    
    def _handle_events(self, data):
        # Validate required fields for events
        required_fields = ['user_id', 'device_name', 'type', 'data', 'timestamp']
        
        for field in required_fields:
            if field not in data:
                print(f"❌ Event missing field: {field}")
                self.send_response(400)  # Bad Request
                self.end_headers()
                return
        
        # Validate event data structure
        if not isinstance(data['data'], dict):
            print("❌ Event data must be a dictionary")
            self.send_response(400)
            self.end_headers()
            return
        
        # Success - store the event
        TestHTTPHandler.events_received.append({
            'user_id': data['user_id'],
            'device_name': data['device_name'],
            'type': data['type'],
            'event_type': data['data'].get('event_type', 'unknown')
        })
        
        print(f"✅ Event received: {data['data'].get('event_type', 'unknown')} from {data['user_id']}")
        self.send_response(200)
        self.end_headers()
    
    def _handle_gamestate_get(self):
        # Return a mock GameUpdate response
        from lumotag_events import PlayerStatus
        
        # Create mock game state with some players
        mock_gamestate = {
            "players": [
                {
                    "health": 100,
                    "ammo": 30,
                    "tag_id": "player1",
                    "display_name": "Test Player 1"
                },
                {
                    "health": 75,
                    "ammo": 15,
                    "tag_id": "player2", 
                    "display_name": "Test Player 2"
                }
            ]
        }
        
        print(f"✅ Gamestate requested - returning {len(mock_gamestate['players'])} players")
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(mock_gamestate).encode())

# Start test server
server_port = 8899
server = HTTPServer(('127.0.0.1', server_port), TestHTTPHandler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()
time.sleep(0.1)  # Let server start

print(f"🌐 Test server started on http://127.0.0.1:{server_port}")

# Mock shared memory setup
def create_test_image_with_id(width=320, height=240):
    """Create a test grayscale image with embedded ID"""
    from factory import create_image_id
    
    img = np.zeros((height, width), dtype=np.uint8)
    # Add some pattern
    img[50:150, 50:150] = 128  # Gray square
    img[100:120, 100:120] = 255  # White center
    
    # Embed ID in first row
    img_id = create_image_id()
    img[0, 0:img_id.shape[0]] = img_id
    
    return img

class MockSharedMem:
    def __init__(self, data):
        self.buf = memoryview(bytearray(data))

test_img = create_test_image_with_id()
img_bytes = test_img.tobytes()
sharedmem_buffs = {0: MockSharedMem(img_bytes)}

def safe_mem_details_func():
    return SharedMem_ImgTicket(
        index=0,
        res=(240, 320),  # height, width
        buf_size=len(img_bytes),
        id=1
    )

try:
    # Test 1: Create HTTPImageComms instance
    print("\n📝 Test 1: Creating HTTPImageComms instance...")
    
    http_comms = HTTPComms(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        images_url=f"http://127.0.0.1:{server_port}/api/v1/images/upload",
        events_url=f"http://127.0.0.1:{server_port}/api/v1/events",
        gamestate_url=f"http://127.0.0.1:{server_port}/api/v1/gamestate",
        OS_friendly_name="test_gun",
        user_id="test_player",
        upload_timeout=1.0
    )
    
    print("✅ HTTPImageComms created successfully")
    time.sleep(0.2)  # Let threads start
    
    # Test 2: Happy Path - Image Upload
    print("\n📝 Test 2: Image Upload (Happy Path)...")
    
    initial_images = len(TestHTTPHandler.images_received)
    
    # Trigger capture and upload
    http_comms.trigger_capture()
    time.sleep(0.1)  # Let capture process
    
    # Get captured image IDs and upload one
    captured_images = list(http_comms.ImageMem.keys())
    if not captured_images:
        raise AssertionError("No images were captured")
    
    http_comms.upload_image_by_id(captured_images[0])
    time.sleep(0.5)  # Let upload process
    
    if len(TestHTTPHandler.images_received) <= initial_images:
        raise AssertionError("Image upload failed - server received no images")
    
    latest_upload = TestHTTPHandler.images_received[-1]
    if latest_upload['user_id'] != 'test_player':
        raise AssertionError(f"Wrong user_id: expected 'test_player', got '{latest_upload['user_id']}'")
    
    if latest_upload['device_name'] != 'test_gun':
        raise AssertionError(f"Wrong device_name: expected 'test_gun', got '{latest_upload['device_name']}'")
    
    print(f"✅ Image upload successful: {latest_upload['image_id']} ({latest_upload['size']} bytes)")
    
    # Test 3: Happy Path - Event Sending
    print("\n📝 Test 3: Event Sending (Happy Path)...")
    
    initial_events = len(TestHTTPHandler.events_received)
    
    # Send a valid event
    test_event = PlayerStatus(
        health=75,
        ammo=25,
        tag_id="test_player",
        display_name="Test Player"
    )
    
    http_comms.send_event(test_event)
    time.sleep(0.3)  # Let event process
    
    if len(TestHTTPHandler.events_received) <= initial_events:
        raise AssertionError("Event sending failed - server received no events")
    
    latest_event = TestHTTPHandler.events_received[-1]
    if latest_event['user_id'] != 'test_player':
        raise AssertionError(f"Wrong user_id in event: expected 'test_player', got '{latest_event['user_id']}'")
    
    if latest_event['event_type'] != 'PlayerStatus':
        raise AssertionError(f"Wrong event_type: expected 'PlayerStatus', got '{latest_event['event_type']}'")
    
    print(f"✅ Event sending successful: {latest_event['event_type']} from {latest_event['user_id']}")
    
    # Test 4: Malformed Event (should raise ValueError)
    print("\n📝 Test 4: Invalid Event Type (should fail validation)...")
    
    class InvalidEvent:
        def __init__(self):
            self.data = "not a pydantic model"
    
    try:
        http_comms.send_event(InvalidEvent())
        raise AssertionError("Should have raised ValueError for invalid event type")
    except ValueError as e:
        if "Event must be one of" not in str(e):
            raise AssertionError(f"Wrong error message: {e}")
        print("✅ Invalid event type correctly rejected")
    
    # Test 5: HTTPImageComms Error Handling
    print("\n📝 Test 5: HTTPImageComms handles server errors gracefully...")
    
    # Create a server that returns errors for testing
    class ErrorHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress logging
        
        def do_POST(self):
            # Always return 500 Internal Server Error
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Server Error for Testing")
    
    # Start error server on different port
    error_port = 8898
    error_server = HTTPServer(('127.0.0.1', error_port), ErrorHTTPHandler)
    error_thread = threading.Thread(target=error_server.serve_forever, daemon=True)
    error_thread.start()
    time.sleep(0.1)
    
    try:
        # Create HTTPImageComms pointing to error server
        error_comms = HTTPComms(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            images_url=f"http://127.0.0.1:{error_port}/api/v1/images/upload",
            events_url=f"http://127.0.0.1:{error_port}/api/v1/events",
            gamestate_url=f"http://127.0.0.1:{error_port}/api/v1/gamestate",
            OS_friendly_name="error_test_gun",
            user_id="error_test_player",
            upload_timeout=0.2  # Short timeout for faster test
        )
        
        time.sleep(0.1)
        
        # Test image upload to failing server - should not crash
        error_comms.trigger_capture()
        time.sleep(0.1)
        
        captured_images = list(error_comms.ImageMem.keys())
        if captured_images:
            # This should fail gracefully (not crash the thread)
            error_comms.upload_image_by_id(captured_images[0])
            time.sleep(0.5)  # Let it try and fail
            
            # Image should still be in memory (not removed due to failure)
            remaining_images = list(error_comms.ImageMem.keys())
            if not remaining_images:
                raise AssertionError("Image was removed despite upload failure")
            
            print("✅ HTTPImageComms handles server errors gracefully (image retained)")
        
        # Test event sending to failing server - should not crash
        test_event = PlayerStatus(
            health=50,
            ammo=10,
            tag_id="error_test_player", 
            display_name="Error Test Player"
        )
        
        # This should fail gracefully (not crash the thread)
        error_comms.send_event(test_event)
        time.sleep(0.3)  # Let it try and fail
        
        # Check that threads are still alive after errors
        if not error_comms._upload_thread.is_alive():
            raise AssertionError("Upload thread died due to server error")
        
        if not error_comms._events_thread.is_alive():
            raise AssertionError("Events thread died due to server error")
            
        if not error_comms._gamestate_thread.is_alive():
            raise AssertionError("Gamestate thread died due to server error")
        
        print("✅ HTTPImageComms threads survive server errors")
        
    finally:
        error_server.shutdown()
    
    print("✅ HTTPImageComms error handling test completed")
    
    # Test 6: Non-existent server (Connection Refused)
    print("\n📝 Test 6: Non-existent server handling...")
    
    # Use a port that definitely has no server running
    nonexistent_port = 9999
    
    try:
        # Create HTTPImageComms pointing to non-existent server
        broken_comms = HTTPComms(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            images_url=f"http://127.0.0.1:{nonexistent_port}/api/v1/images/upload",
            events_url=f"http://127.0.0.1:{nonexistent_port}/api/v1/events",
            gamestate_url=f"http://127.0.0.1:{nonexistent_port}/api/v1/gamestate",
            OS_friendly_name="broken_test_gun",
            user_id="broken_test_player",
            upload_timeout=0.2  # Short timeout for faster test
        )
        
        time.sleep(0.1)
        
        # Test connection state - should start as connected (optimistic)
        if not broken_comms.is_connected():
            raise AssertionError("HTTPComms should start in connected state")
        
        print("✅ HTTPComms starts optimistically connected")
        
        # Test image upload to non-existent server
        broken_comms.trigger_capture()
        time.sleep(0.1)
        
        captured_images = list(broken_comms.ImageMem.keys())
        if captured_images:
            print(f"📤 Attempting upload to non-existent server...")
            
            # This should fail with connection refused
            broken_comms.upload_image_by_id(captured_images[0])
            time.sleep(1.0)  # Give it time to fail and apply delay
            
            # Connection state should now be disconnected
            if broken_comms.is_connected():
                raise AssertionError("HTTPComms should be disconnected after connection failure")
            
            print("✅ HTTPComms correctly detects connection failure")
            
            print("✅ Connection state correctly tracked")
            
            # Image should still be in memory (not removed due to failure)
            remaining_images = list(broken_comms.ImageMem.keys())
            if not remaining_images:
                raise AssertionError("Image was removed despite connection failure")
            
            print("✅ Image retained in memory after connection failure")
        
        # Test event sending to non-existent server
        test_event = PlayerStatus(
            health=100,
            ammo=30,
            tag_id="broken_test_player", 
            display_name="Broken Test Player"
        )
        
        print(f"📨 Attempting event send to non-existent server...")
        broken_comms.send_event(test_event)
        time.sleep(1.0)  # Give it time to fail and apply delay
        
        # Should still be disconnected
        if broken_comms.is_connected():
            raise AssertionError("HTTPComms should remain disconnected after event failure")
        
        print("✅ HTTPComms remains disconnected after event failure")
        
        # Check that threads are still alive after connection failures
        if not broken_comms._upload_thread.is_alive():
            raise AssertionError("Upload thread died due to connection failure")
        
        if not broken_comms._events_thread.is_alive():
            raise AssertionError("Events thread died due to connection failure")
            
        if not broken_comms._gamestate_thread.is_alive():
            raise AssertionError("Gamestate thread died due to connection failure")
        
        print("✅ HTTPComms threads survive connection failures")
        
        # Test queue sizes
        upload_queue_size = broken_comms.get_upload_queue_size()
        events_queue_size = broken_comms.get_events_queue_size()
        
        print(f"✅ Queue sizes accessible: upload={upload_queue_size}, events={events_queue_size}")
        
    except Exception as e:
        print(f"💥 Non-existent server test failed: {e}")
        raise
    
    print("✅ Non-existent server handling test completed")
    
    # Test 7: Gamestate Retrieval
    print("\n📝 Test 7: Gamestate Retrieval (Client-side)...")
    
    # Give the gamestate thread time to poll and store data
    time.sleep(1.0)  # Wait for at least one gamestate poll cycle
    
    # Get the latest gamestate from the client
    latest_gamestate = http_comms.get_latest_gamestate()
    
    if latest_gamestate is None:
        raise AssertionError("No gamestate retrieved from server")
    
    # Verify it's a proper GameUpdate object
    from lumotag_events import GameUpdate
    if not isinstance(latest_gamestate, GameUpdate):
        raise AssertionError(f"Expected GameUpdate object, got {type(latest_gamestate)}")
    
    # Verify it matches the mock server data
    expected_players = 2
    if len(latest_gamestate.players) != expected_players:
        raise AssertionError(f"Expected {expected_players} players, got {len(latest_gamestate.players)}")
    
    # Check specific player data matches mock server
    player1 = latest_gamestate.players[0]
    if player1.tag_id != "player1" or player1.health != 100 or player1.ammo != 30:
        raise AssertionError(f"Player 1 data doesn't match mock server: {player1}")
    
    player2 = latest_gamestate.players[1]  
    if player2.tag_id != "player2" or player2.health != 75 or player2.ammo != 15:
        raise AssertionError(f"Player 2 data doesn't match mock server: {player2}")
    
    print(f"✅ Gamestate retrieved successfully: {len(latest_gamestate.players)} players")
    print(f"   Player 1: {player1.display_name} (HP: {player1.health}, Ammo: {player1.ammo})")
    print(f"   Player 2: {player2.display_name} (HP: {player2.health}, Ammo: {player2.ammo})")
    
    print("✅ Gamestate retrieval test completed")
    
    # Test Summary
    print("\n🎉 ALL TESTS PASSED!")
    print(f"📊 Test Results:")
    print(f"   📤 Images uploaded: {len(TestHTTPHandler.images_received)}")
    print(f"   📨 Events sent: {len(TestHTTPHandler.events_received)}")
    print(f"   ✅ HTTP status codes validated")
    print(f"   ✅ Data validation working")
    print(f"   ✅ User identification working")
    print(f"   ✅ Connection state tracking working")
    print(f"   ✅ Non-existent server handling working") 
    print(f"   ✅ Gamestate retrieval working")
    
except Exception as e:
    print(f"\n💥 TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

finally:
    # Cleanup
    server.shutdown()
    print("\n🧹 Test server stopped")
    print("✨ Tests completed!")
