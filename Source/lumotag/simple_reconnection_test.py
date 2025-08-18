#!/usr/bin/env python3
"""
Simple WebSocket Reconnection Test

Tests the core reconnection functionality without complex server restarts.
"""

import time
import threading
import subprocess
import sys
from comms import WebSocketComms
from my_collections import SharedMem_ImgTicket
import numpy as np

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

def test_reconnection_behavior():
    """Test reconnection behavior with a non-existent server"""
    print("🧪 Simple Reconnection Test")
    print("="*50)
    
    sharedmem_buffs, safe_mem_details_func, embedded_id = create_test_setup()
    
    # Create uploader pointing to non-existent server
    print("1️⃣ Creating uploader with unreachable server")
    uploader = WebSocketComms(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url="ws://localhost:65533",  # Non-existent server
        OS_friendly_name="test_reconnection"
    )
    
    print("2️⃣ Capturing and queuing images")
    for i in range(3):
        uploader.trigger_capture()
        time.sleep(0.05)
        uploader.upload_image_by_id(embedded_id)
        print(f"   📦 Queued image {i+1}")
    
    print("3️⃣ Checking queue status")
    stored_ids = uploader.get_stored_image_ids()
    print(f"   📊 Images in memory: {len(stored_ids)}")
    
    print("4️⃣ Monitoring thread health over time")
    for check in range(5):
        time.sleep(2)
        try:
            uploader.raise_thread_error_if_any()
            print(f"   ✅ Health check {check+1}: All threads healthy")
        except RuntimeError as e:
            print(f"   ❌ Health check {check+1}: {e}")
            break
    
    print("5️⃣ Final status")
    final_stored = uploader.get_stored_image_ids()
    print(f"   📊 Final images in memory: {len(final_stored)}")
    print("   ℹ️  Images should remain queued until server becomes available")
    
    return len(final_stored) > 0

def test_thread_restart_timing():
    """Test how quickly threads restart after failure"""
    print("\n🧪 Thread Restart Timing Test")
    print("="*50)
    
    sharedmem_buffs, safe_mem_details_func, embedded_id = create_test_setup()
    
    uploader = WebSocketComms(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url="ws://localhost:65532",  # Another non-existent server
        OS_friendly_name="test_timing"
    )
    
    print("1️⃣ Monitoring thread restart frequency")
    restart_count = 0
    
    for minute in range(3):  # Monitor for 3 minutes
        print(f"   ⏱️  Minute {minute + 1}:")
        
        for check in range(6):  # Check every 10 seconds
            time.sleep(10)
            try:
                uploader.raise_thread_error_if_any()
                print(f"      ✅ Check {check+1}: Threads healthy")
            except Exception as e:
                restart_count += 1
                print(f"      🔄 Check {check+1}: Thread restart #{restart_count}")
        
        # Queue some work
        uploader.trigger_capture()
        uploader.upload_image_by_id(embedded_id)
    
    print(f"   📊 Total restarts in 3 minutes: {restart_count}")
    print(f"   📊 Average restart interval: {180/max(restart_count,1):.1f} seconds")
    
    return restart_count > 0

def main():
    """Run simple reconnection tests"""
    print("🔄 WebSocket Reconnection Behavior Tests")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Message Queuing", test_reconnection_behavior()))
        results.append(("Thread Restart Timing", test_thread_restart_timing()))
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    
    # Results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Reconnection behavior is working correctly!")
    else:
        print("⚠️  Some reconnection issues detected.")

if __name__ == "__main__":
    main()
