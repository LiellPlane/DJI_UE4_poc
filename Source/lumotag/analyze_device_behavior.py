#!/usr/bin/env python3
"""
Device Behavior Analyzer

Analyzes patterns in device connection issues to identify root causes:
- Memory leaks on device
- Network buffer overflows  
- WiFi signal issues
- Device power management
- Periodic reconnection patterns
"""
import time
from datetime import datetime, timedelta


def analyze_device_patterns():
    """Analyze common device-side issues that cause image flow to stop"""
    
    print("🔍 DEVICE-SIDE ISSUE ANALYSIS")
    print("=" * 50)
    
    print("\n📱 COMMON DEVICE ISSUES THAT STOP IMAGE FLOW:")
    
    print("\n1. 🧠 MEMORY ISSUES:")
    print("   - Device runs out of RAM after 30+ seconds")
    print("   - Image buffers accumulate without cleanup")
    print("   - Garbage collection pauses block sending")
    print("   ✅ Solution: Add memory monitoring, smaller buffers")
    
    print("\n2. 📶 NETWORK BUFFER OVERFLOW:")
    print("   - Device sends faster than network can handle")
    print("   - TCP send buffer fills up and blocks")
    print("   - WebSocket frame queuing issues")
    print("   ✅ Solution: Add flow control, reduce send rate")
    
    print("\n3. 🔋 POWER MANAGEMENT:")
    print("   - Device enters power-save mode")
    print("   - WiFi adapter reduces power")
    print("   - CPU throttling affects image processing")
    print("   ✅ Solution: Disable power saving, keep-alive messages")
    
    print("\n4. 🌐 WIFI ISSUES:")
    print("   - Signal strength degrades over time")
    print("   - Router drops inactive connections")
    print("   - Channel congestion increases")
    print("   ✅ Solution: Connection health checks, auto-reconnect")
    
    print("\n5. 🔄 WEBSOCKET ISSUES:")
    print("   - Connection silently drops (no error)")
    print("   - Send queue backs up without notification")
    print("   - Ping/pong timeouts")
    print("   ✅ Solution: Heartbeat messages, connection monitoring")
    
    print("\n6. 📷 IMAGE PROCESSING ISSUES:")
    print("   - Camera driver hangs or crashes")
    print("   - Image encoding takes too long")
    print("   - File I/O blocks the send thread")
    print("   ✅ Solution: Async processing, timeouts")


def suggest_device_fixes():
    """Suggest specific fixes for device-side issues"""
    
    print("\n🔧 RECOMMENDED DEVICE-SIDE FIXES:")
    print("=" * 40)
    
    print("\n1. ADD CONNECTION HEALTH MONITORING:")
    print("""
    # Check if websocket is still connected
    if websocket.closed:
        print("Connection lost - reconnecting...")
        await reconnect()
    
    # Send periodic ping messages
    await websocket.ping()
    """)
    
    print("\n2. IMPLEMENT SEND QUEUE MONITORING:")
    print("""
    # Check send buffer size before sending
    if len(websocket.send_queue) > 10:
        print("Send queue backing up - slowing down")
        await asyncio.sleep(0.1)
    """)
    
    print("\n3. ADD MEMORY MONITORING:")
    print("""
    import psutil
    
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        print("Memory usage high - triggering cleanup")
        gc.collect()
    """)
    
    print("\n4. IMPLEMENT AUTO-RECONNECTION:")
    print("""
    async def send_with_retry(websocket, message):
        for attempt in range(3):
            try:
                await websocket.send(message)
                return
            except websockets.ConnectionClosed:
                print(f"Reconnection attempt {attempt + 1}")
                websocket = await reconnect()
    """)
    
    print("\n5. ADD FLOW CONTROL:")
    print("""
    # Limit send rate based on server response
    last_send_time = time.time()
    
    if time.time() - last_send_time < 0.1:  # Max 10 images/sec
        await asyncio.sleep(0.1 - (time.time() - last_send_time))
    """)


def create_device_test_checklist():
    """Create a checklist for testing device behavior"""
    
    print("\n📋 DEVICE TESTING CHECKLIST:")
    print("=" * 30)
    
    checklist = [
        "□ Monitor device memory usage during image sending",
        "□ Check WiFi signal strength over time",
        "□ Test with different image send rates (1/sec, 5/sec, 10/sec)",
        "□ Monitor websocket connection state",
        "□ Check device logs for errors/warnings",
        "□ Test with power saving disabled",
        "□ Monitor network buffer usage",
        "□ Test reconnection after network interruption",
        "□ Check for memory leaks in image processing",
        "□ Monitor CPU usage during image capture/send"
    ]
    
    for item in checklist:
        print(f"   {item}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Run connection_monitor.py to watch your device behavior")
    print(f"   2. Point your device to ws://localhost:8767")
    print(f"   3. Watch for patterns in disconnections and flow stops")
    print(f"   4. Check device logs when flow stops")
    print(f"   5. Implement suggested fixes based on patterns found")


if __name__ == "__main__":
    analyze_device_patterns()
    suggest_device_fixes()
    create_device_test_checklist()
    
    print(f"\n🚀 TO START MONITORING:")
    print(f"   python connection_monitor.py")
    print(f"   (Then point your device to ws://localhost:8767)")
