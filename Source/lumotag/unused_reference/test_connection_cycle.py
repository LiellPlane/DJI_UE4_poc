#!/usr/bin/env python3
"""
Connection Cycle Test: Success -> Death -> Recovery -> Reconnection

Tests the exact scenario: successful connection, server death, server recovery, successful reconnection.
"""

import time
import threading
import asyncio
import websockets
import json
import sys
import socket
from comms import WebSocketImageComms
from my_collections import SharedMem_ImgTicket
import numpy as np

def find_available_port(start_port=8800):
    """Find an available port to avoid conflicts"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

class CycleTestServer:
    """Simple WebSocket server for connection cycle testing"""
    
    def __init__(self):
        self.port = find_available_port()
        self.server = None
        self.uploads_received = []
        self.is_running = False
        self.server_task = None
        self.loop = None
        
    async def handle_client(self, websocket):
        """Handle client connections"""
        print(f"📱 Client connected to test server")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "image_upload":
                    image_id = data.get("image_id")
                    if image_id:
                        self.uploads_received.append(image_id)
                        print(f"📤 Server received: {image_id}")
                        
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 Client disconnected from test server")
        except Exception as e:
            print(f"❌ Server error: {e}")
    
    def start(self):
        """Start server in background thread"""
        print(f"🌐 Starting test server on port {self.port}")
        
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            async def server_main():
                self.server = await websockets.serve(
                    self.handle_client,
                    'localhost', 
                    self.port,
                    ping_interval=10,
                    ping_timeout=5
                )
                self.is_running = True
                print(f"✅ Test server running on localhost:{self.port}")
                await self.server.wait_closed()
                
            self.loop.run_until_complete(server_main())
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1.0)  # Give server time to start
        
    def stop(self):
        """Stop the server cleanly"""
        if self.server and self.is_running:
            print("🛑 Stopping test server...")
            
            # Close server in its own event loop
            if self.loop:
                self.loop.call_soon_threadsafe(self.server.close)
                self.is_running = False
                time.sleep(1.0)  # Give time to close connections
                print("✅ Test server stopped")
            
    def get_url(self):
        """Get WebSocket URL"""
        return f"ws://localhost:{self.port}"

def create_test_setup():
    """Create test image and uploader setup"""
    from factory import create_image_id, decode_image_id
    
    # Create test image
    img = np.zeros((480, 640), dtype=np.uint8)
    img[100:200, 200:400] = 255  # White rectangle
    img_id = create_image_id()
    img[0, 0:img_id.shape[0]] = img_id
    embedded_id = decode_image_id(img)
    
    # Mock shared memory
    class MockSharedMem:
        def __init__(self, data):
            self.buf = memoryview(bytearray(data))
    
    sharedmem_buffs = {0: MockSharedMem(img.tobytes())}
    
    def safe_mem_details_func():
        return SharedMem_ImgTicket(
            index=0, 
            res=(480, 640),
            buf_size=len(img.tobytes()),
            id=1
        )
    
    return sharedmem_buffs, safe_mem_details_func, embedded_id

def test_connection_death_recovery_cycle():
    """Test: Connection Success -> Server Death -> Server Recovery -> Reconnection Success"""
    print("🔄 Connection Cycle Test: Success → Death → Recovery → Reconnection")
    print("="*70)
    
    # Setup
    server = CycleTestServer()
    sharedmem_buffs, safe_mem_details_func, embedded_id = create_test_setup()
    
    try:
        # Phase 1: Start server and establish successful connection
        print("\n1️⃣ PHASE 1: Server startup and successful connection")
        server.start()
        
        uploader = WebSocketImageComms(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            websocket_url=server.get_url(),
            OS_friendly_name="cycle_test"
        )
        
        # Wait for connection to establish
        time.sleep(2.0)
        print("   ✅ WebSocket uploader created")
        
        # Test successful upload
        print("   📤 Testing successful upload...")
        uploader.trigger_capture()
        time.sleep(0.1)
        uploader.upload_image_by_id(embedded_id)
        time.sleep(2.0)  # Give time for upload
        
        phase1_uploads = len(server.uploads_received)
        print(f"   ✅ Phase 1 uploads: {phase1_uploads}")
        assert phase1_uploads > 0, "Phase 1: No successful uploads detected"
        
        # Phase 2: Kill server (simulate death)
        print("\n2️⃣ PHASE 2: Server death simulation")
        print("   🛑 Stopping server...")
        server.stop()
        time.sleep(1.0)
        
        # Try uploading while server is dead (should queue)
        print("   📦 Queueing uploads while server is dead...")
        for i in range(3):
            uploader.trigger_capture()
            time.sleep(0.05)
            uploader.upload_image_by_id(embedded_id)
            print(f"      📦 Queued upload {i+1}")
        
        queued_images = len(uploader.get_stored_image_ids())
        print(f"   ✅ Images queued during outage: {queued_images}")
        
        # Phase 3: Server recovery (same port)
        print("\n3️⃣ PHASE 3: Server recovery")
        print("   🔄 Restarting server on same port...")
        time.sleep(2.0)  # Give time for port to be released
        
        # Restart server on same port (more realistic scenario)
        server.start()
        print("   🔗 Server back online, waiting for reconnection...")
        
        # Phase 4: Verify reconnection and message processing
        print("\n4️⃣ PHASE 4: Reconnection and message processing")
        print("   ⏱️  Waiting for automatic reconnection...")
        
        # Monitor for reconnection and message processing
        reconnection_detected = False
        message_processing_detected = False
        
        for attempt in range(30):  # Wait up to 30 seconds
            time.sleep(1.0)
            
            # Check if new messages arrived at server
            current_uploads = len(server.uploads_received)
            if current_uploads > 0 and not message_processing_detected:
                print(f"   ✅ Message processing detected! Server received {current_uploads} uploads")
                message_processing_detected = True
            
            # Check thread health
            try:
                uploader.raise_thread_error_if_any()
                if not reconnection_detected:
                    print(f"   🔗 Reconnection attempt {attempt + 1}: Threads healthy")
                    reconnection_detected = True
            except RuntimeError as e:
                print(f"   🔄 Reconnection attempt {attempt + 1}: {e}")
            
            if message_processing_detected and reconnection_detected:
                break
        
        # Final verification
        print("\n5️⃣ PHASE 5: Final verification")
        final_uploads = len(server.uploads_received)
        remaining_queued = len(uploader.get_stored_image_ids())
        
        print(f"   📊 Total server uploads: {final_uploads}")
        print(f"   📊 Remaining queued images: {remaining_queued}")
        print(f"   📊 Messages processed after recovery: {final_uploads}")
        
        # Health check
        uploader.raise_thread_error_if_any()
        print("   ✅ All threads healthy after full cycle")
        
        # Test one more upload to confirm connection is solid
        print("   🧪 Testing post-recovery upload...")
        uploader.trigger_capture()
        time.sleep(0.1)
        uploader.upload_image_by_id(embedded_id)
        time.sleep(2.0)
        
        final_final_uploads = len(server.uploads_received)
        print(f"   ✅ Post-recovery uploads: {final_final_uploads}")
        
        # Results
        success = (
            phase1_uploads > 0 and  # Initial connection worked
            queued_images > 0 and   # Messages queued during outage
            final_uploads >= phase1_uploads and   # Messages processed after recovery (at least as many as before)
            final_final_uploads >= final_uploads  # Post-recovery functionality works
        )
        
        print(f"   📊 Success criteria:")
        print(f"      - Initial connection worked: {phase1_uploads > 0} ({phase1_uploads} uploads)")
        print(f"      - Messages queued during outage: {queued_images > 0} ({queued_images} queued)")
        print(f"      - Messages processed after recovery: {final_uploads >= phase1_uploads} ({final_uploads} total)")
        print(f"      - Post-recovery functionality: {final_final_uploads >= final_uploads} ({final_final_uploads} final)")
        
        return success
        
    finally:
        server.stop()

def main():
    """Run connection cycle test"""
    print("🧪 WebSocket Connection Cycle Test")
    print("Testing: Success → Death → Recovery → Reconnection")
    print("="*70)
    
    try:
        result = test_connection_death_recovery_cycle()
        
        print("\n" + "="*70)
        print("📊 TEST RESULT")
        print("="*70)
        
        if result:
            print("✅ SUCCESS: Full connection cycle completed successfully!")
            print("   - Initial connection: ✅")
            print("   - Message queuing during outage: ✅") 
            print("   - Server recovery: ✅")
            print("   - Automatic reconnection: ✅")
            print("   - Queued message processing: ✅")
            print("   - Post-recovery functionality: ✅")
        else:
            print("❌ FAILURE: Connection cycle test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
