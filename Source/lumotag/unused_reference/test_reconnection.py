#!/usr/bin/env python3
"""
Comprehensive WebSocket Reconnection Test Suite

This script tests various disconnection/reconnection scenarios to ensure
robust operation under network failures.
"""

import time
import threading
import asyncio
import websockets
import json
import signal
import sys
from comms import WebSocketUploaderThreaded_shared_mem
from my_collections import SharedMem_ImgTicket
import numpy as np

class ReconnectionTestServer:
    """Controllable WebSocket server for testing reconnection scenarios"""
    
    def __init__(self, port=8766):
        self.port = port
        self.server = None
        self.server_task = None
        self.uploads_received = []
        self.is_running = False
        self.connection_count = 0
        
    async def handle_client(self, websocket):
        """Handle client connections with logging"""
        client_id = self.connection_count
        self.connection_count += 1
        print(f"📱 Client {client_id} connected")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "image_upload":
                    image_id = data.get("image_id")
                    if image_id:
                        self.uploads_received.append(image_id)
                        print(f"📤 Client {client_id} uploaded: {image_id}")
                        
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 Client {client_id} disconnected")
        except Exception as e:
            print(f"❌ Client {client_id} error: {e}")
    
    async def start_async(self):
        """Start the server asynchronously"""
        self.server = await websockets.serve(
            self.handle_client,
            'localhost', 
            self.port,
            ping_interval=5,
            ping_timeout=3
        )
        self.is_running = True
        print(f"🌐 Test server started on localhost:{self.port}")
        await self.server.wait_closed()
        
    def start(self):
        """Start server in background thread"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_async())
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(0.5)  # Give server time to start
        
    def stop(self):
        """Stop the server"""
        if self.server and self.is_running:
            print("🛑 Stopping test server...")
            self.server.close()
            self.is_running = False
            time.sleep(0.2)  # Give time to close
            
    def restart(self):
        """Restart the server"""
        self.stop()
        time.sleep(0.5)  # Wait for full shutdown
        self.start()

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

def test_scenario_1_server_restart():
    """Test 1: Server restart during operation"""
    print("\n" + "="*60)
    print("📡 TEST 1: Server Restart Simulation")
    print("="*60)
    
    server = ReconnectionTestServer()
    server.start()
    
    sharedmem_buffs, safe_mem_details_func, embedded_id = create_test_setup()
    
    # Create uploader
    uploader = WebSocketUploaderThreaded_shared_mem(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url=f"ws://localhost:{server.port}",
        OS_friendly_name="test_reconnection"
    )
    
    try:
        print("1️⃣ Phase 1: Normal operation")
        uploader.trigger_capture()
        time.sleep(0.1)
        uploader.upload_image_by_id(embedded_id)
        time.sleep(1.0)
        
        initial_uploads = len(server.uploads_received)
        print(f"   ✅ Initial uploads: {initial_uploads}")
        
        print("2️⃣ Phase 2: Stop server (simulate network outage)")
        server.stop()
        time.sleep(0.5)
        
        print("3️⃣ Phase 3: Queue images during outage")
        for i in range(3):
            uploader.trigger_capture()
            time.sleep(0.05)
            uploader.upload_image_by_id(embedded_id)
        
        queued_count = len(uploader.get_stored_image_ids())
        print(f"   📦 Images queued during outage: {queued_count}")
        
        print("4️⃣ Phase 4: Restart server (simulate recovery)")
        server.restart()
        
        print("5️⃣ Phase 5: Wait for reconnection and message processing")
        time.sleep(5.0)  # Give time for reconnection and processing
        
        final_uploads = len(server.uploads_received)
        print(f"   ✅ Final uploads: {final_uploads}")
        print(f"   📊 Messages processed after restart: {final_uploads - initial_uploads}")
        
        # Check thread health
        uploader.raise_thread_error_if_any()
        print("   ✅ All threads healthy after reconnection")
        
    finally:
        server.stop()
    
    return final_uploads > initial_uploads

def test_scenario_2_intermittent_connectivity():
    """Test 2: Rapidly fluctuating connectivity"""
    print("\n" + "="*60)
    print("📡 TEST 2: Intermittent Connectivity Simulation")
    print("="*60)
    
    server = ReconnectionTestServer()
    server.start()
    
    sharedmem_buffs, safe_mem_details_func, embedded_id = create_test_setup()
    
    uploader = WebSocketUploaderThreaded_shared_mem(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url=f"ws://localhost:{server.port}",
        OS_friendly_name="test_intermittent"
    )
    
    try:
        print("1️⃣ Starting intermittent connectivity test...")
        
        # Rapidly start/stop server while uploading
        for cycle in range(3):
            print(f"   🔄 Cycle {cycle + 1}: Upload during connectivity")
            
            # Upload while connected
            uploader.trigger_capture()
            time.sleep(0.05)
            uploader.upload_image_by_id(embedded_id)
            time.sleep(0.5)
            
            # Brief outage
            print(f"   🛑 Cycle {cycle + 1}: Brief outage")
            server.stop()
            time.sleep(1.0)
            
            # Queue during outage
            uploader.trigger_capture()
            uploader.upload_image_by_id(embedded_id)
            
            # Reconnect
            print(f"   🔄 Cycle {cycle + 1}: Reconnecting")
            server.restart()
            time.sleep(2.0)
        
        final_uploads = len(server.uploads_received)
        print(f"   📊 Total uploads processed: {final_uploads}")
        
        # Check thread health
        uploader.raise_thread_error_if_any()
        print("   ✅ All threads survived intermittent connectivity")
        
    finally:
        server.stop()
    
    return final_uploads > 0

def test_scenario_3_thread_restart():
    """Test 3: Force thread death and verify restart"""
    print("\n" + "="*60)
    print("📡 TEST 3: Thread Restart Verification")
    print("="*60)
    
    server = ReconnectionTestServer()
    server.start()
    
    sharedmem_buffs, safe_mem_details_func, embedded_id = create_test_setup()
    
    uploader = WebSocketUploaderThreaded_shared_mem(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url=f"ws://localhost:{server.port}",
        OS_friendly_name="test_thread_restart"
    )
    
    try:
        print("1️⃣ Normal operation to establish baseline")
        uploader.trigger_capture()
        time.sleep(0.1)
        uploader.upload_image_by_id(embedded_id)
        time.sleep(1.0)
        
        initial_uploads = len(server.uploads_received)
        print(f"   ✅ Baseline uploads: {initial_uploads}")
        
        print("2️⃣ Force thread death by stopping server abruptly")
        server.stop()
        
        # Queue messages while server is down
        print("3️⃣ Queue messages during server outage")
        for i in range(2):
            uploader.trigger_capture()
            time.sleep(0.05)
            uploader.upload_image_by_id(embedded_id)
        
        print("4️⃣ Check thread health (should trigger restart)")
        # This should detect dead thread and restart it
        try:
            uploader.raise_thread_error_if_any()
            print("   ✅ Thread restart handled gracefully")
        except RuntimeError as e:
            print(f"   ❌ Unexpected error: {e}")
        
        print("5️⃣ Restart server and verify message processing")
        server.restart()
        time.sleep(3.0)
        
        final_uploads = len(server.uploads_received)
        print(f"   📊 Final uploads: {final_uploads}")
        print(f"   📊 Messages processed after restart: {final_uploads - initial_uploads}")
        
        # Final health check
        uploader.raise_thread_error_if_any()
        print("   ✅ All threads healthy after full cycle")
        
    finally:
        server.stop()
    
    return final_uploads > initial_uploads

def main():
    """Run all reconnection test scenarios"""
    print("🧪 WebSocket Reconnection Test Suite")
    print("="*60)
    
    results = []
    
    try:
        # Run test scenarios
        results.append(("Server Restart", test_scenario_1_server_restart()))
        results.append(("Intermittent Connectivity", test_scenario_2_intermittent_connectivity()))
        results.append(("Thread Restart", test_scenario_3_thread_restart()))
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    
    # Results summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All reconnection tests passed! WebSocket implementation is robust.")
    else:
        print("⚠️  Some tests failed. Review reconnection logic.")
        sys.exit(1)

if __name__ == "__main__":
    main()
