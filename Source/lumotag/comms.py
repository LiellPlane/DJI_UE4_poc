
# WEBSOCKET COMMUNICATION STRATEGY FOR LUMOTAG
# =============================================

# PROBLEM: WebSockets are bidirectional but we want send-only for image streams
# - Server broadcasts game state to all connected devices
# - If we don't read incoming messages, TCP buffers fill up and cause issues
# - Need to maintain WebSocket standard for consistency across the system

# SOLUTION: Parallel Async Handlers
# =================================
# Use asyncio.gather() to run send and receive handlers simultaneously:
# 
# await asyncio.gather(
#     self._send_handler(websocket, send_queue),     # Sends our images
#     self._discard_handler(websocket),              # Reads and throws away incoming
#     return_exceptions=True
# )
#
# This runs both handlers in parallel - neither blocks the other
# When one is waiting for I/O, the other runs (event loop interleaving)

# KEY INSIGHTS:
# =============
# 1. Async/await yields control - doesn't block the main thread
# 2. asyncio.gather() runs tasks concurrently, not sequentially  
# 3. WebSocket reads and writes can happen simultaneously
# 4. Discarding incoming data prevents buffer buildup without blocking sends

# IMPLEMENTATION PATTERN:
# ======================
# 1. Create separate Process for each WebSocket (like analyse_lumotag.py)
# 2. Use Queue for communication between main process and WebSocket process
# 3. In WebSocket process, use asyncio.gather() for parallel send/receive
# 4. Send handler: get data from queue, send via websocket.send()
# 5. Discard handler: async for message in websocket, do nothing with it
# 6. Main process just copies data and puts in queue (sub-millisecond)

# ARCHITECTURE:
# =============
# - ImageWebSocket: Send-only, discards incoming (for camera streams)
# - GameStateWebSocket: Bidirectional, processes incoming (for game events)
# - LumotagComms: Manager class that coordinates multiple WebSockets
#
# Each WebSocket runs in its own Process to avoid GIL blocking
# Main loop stays responsive - just copies and queues data

# WHY THIS WORKS:
# ===============
# - WebSocket protocol compliance (handles control frames properly)
# - No buffer buildup (incoming data immediately discarded)
# - No blocking (parallel handlers via asyncio)
# - Real-time friendly (drops frames if queue full)
# - Follows existing pattern (same as ImageAnalyser_shared_mem)

# INTEGRATION:
# ============
# In lumogun.py main loop:
# 1. Create LumotagComms instance after other components
# 2. Call send_trigger_event() when trigger pressed (instant, non-blocking)
# 3. Call get_game_updates() each loop to check for server updates
# 4. Process any received game state updates

# TCP OPTIMIZATION (optional):
# ===========================
# - Set small receive buffer: sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
# - Disable Nagle: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# - Large send buffer: sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
# - Disable ping/pong: ping_interval=None, ping_timeout=None

# NEXT STEPS:
# ===========
# 1. Implement the classes above with proper error handling
# 2. Add logging for connection status and errors
# 3. Test with your game server endpoints
# 4. Integrate into lumogun.py main loop
# 5. Monitor performance and adjust queue sizes as needed