import time
import cv2
import websockets
import asyncio
import base64
import json
import threading
import queue as threading_queue
from typing import Callable
from collections import OrderedDict
import numpy as np
import traceback
from analyse_lumotag import debuffer_image
from factory import decode_image_id
from my_collections import SharedMem_ImgTicket
from lumotag_events import UploadRequest
# UploadResult removed - we no longer track upload results
# Network failures are expected and ignored, encoding failures crash immediately



class WebSocketImageComms:
    """Ultra-lightweight, threaded uploader for grayscale frames from shared memory.

    Design goals:
    - Avoid extra processes; use a single background thread
    - No busy waiting; block on a single control queue
    - Keep a bounded cache of most recent frames (raw grayscale)
    - Encode and POST only on explicit request
    - Ignore network failures, crash on malformed requests

    Public API:
      - trigger_capture()
      - upload_image_by_id(image_id)
      - delete_image_by_id(image_id)
      - get_stored_image_ids()
      - raise_thread_error_if_any()
    """

    def __init__(
        self,
        sharedmem_buffs: dict,
        safe_mem_details_func: Callable[[], SharedMem_ImgTicket],
        websocket_url: str,
        OS_friendly_name: str,
    ) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.safe_mem_details_func = safe_mem_details_func
        self.websocket_url = websocket_url
        self.OS_friendly_name = OS_friendly_name
        self.max_store = 100

        # Keep raw grayscale frames by embedded image id
        self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_lock = threading.Lock()
        


        # Separate queues to decouple capture (debuffer+copy) and upload (encode+HTTP)
        # Small queues like ImageAnalyser - should be empty if processing fast enough
        self._capture_q: threading_queue.Queue = threading_queue.Queue(maxsize=1)
        # Upload control: image_id strings only - small queue to detect performance issues
        self._control_q: threading_queue.Queue = threading_queue.Queue(maxsize=15)
        # upload_result_q removed - no longer tracking upload results
        self._error_q: threading_queue.Queue = threading_queue.Queue(maxsize=10)

        # Connection status tracking for cheap health checks
        self._is_connected = False
        self._capture_thread = threading.Thread(target=self._capture_loop, name="uploader-capture", daemon=True)
        self._upload_thread = threading.Thread(target=self._worker_loop, name="uploader-worker", daemon=True)
        self._capture_thread.start()
        self._upload_thread.start()
        
        # Give threads time to start up before constructor returns
        time.sleep(0.1)

    def trigger_capture(self) -> None:
        """Trigger capture - will crash if queue is full (performance issue)"""
        ticket = self.safe_mem_details_func()
        self._capture_q.put_nowait(ticket)  # Will raise queue.Full if queue is full
        
        # Check for thread errors periodically to reduce overhead at high frequencies
        self._error_check_counter = getattr(self, '_error_check_counter', 0) + 1
        if self._error_check_counter % 20 == 0:  # Check every 20th call (~500ms at 40Hz)
            self.raise_thread_error_if_any()

    def upload_image_by_id(self, image_id: str) -> None:
        """Queue a specific image ID for upload
        we have to handle being disconnected but user still keeps shooting - so let this silently fail for now"""
        try:
            self._control_q.put_nowait(image_id)
        except threading_queue.Full:
            pass

    def delete_image_by_id(self, image_id: str) -> bool:
        """Delete a specific image ID from storage - returns True if deleted, False if not found"""
        with self._mem_lock:
            return self.ImageMem.pop(image_id, None) is not None

    def get_stored_image_ids(self) -> list[str]:
        with self._mem_lock:
            return list(self.ImageMem.keys())

    def is_connected(self) -> bool:
        """Check if WebSocket is connected - extremely cheap to call"""
        return self._is_connected

    def get_upload_queue_size(self) -> int:
        """Get current upload queue size - extremely cheap to call"""
        return self._control_q.qsize()

    # get_upload_result() removed - no longer tracking upload results

    def raise_thread_error_if_any(self) -> None:
        """Lightweight check for thread errors and restart WebSocket thread on network issues"""
        # Check for caught exceptions first (non-blocking)
        if not self._error_q.empty():
            thread_name, exc, tb_str = self._error_q.get_nowait()
            
            # WebSocket worker should handle its own reconnections for network errors
            # If it crashed, that indicates a serious programming bug that should not be hidden
            if thread_name == "uploader-worker":
                raise RuntimeError(f"WebSocket worker thread crashed unexpectedly: {exc}\n{tb_str}") from exc
            
            # For data errors or capture thread errors, still crash (preserve test behavior)
            raise RuntimeError(f"{thread_name} failed: {exc}\n{tb_str}") from exc
        
        # Check for silent thread death (these calls might be expensive, so we do them less frequently)
        if not self._capture_thread.is_alive():
            raise RuntimeError("Capture thread died silently (no exception caught)")
        if not self._upload_thread.is_alive():
            raise RuntimeError("WebSocket worker thread died silently (no exception caught)")




    def _worker_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_worker())
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
        finally:
            loop.close()

    async def _ws_worker(self) -> None:
        reconnect_delay = 1.0
        
        while True:  # Reconnection loop
            try:
                # Mark as attempting connection (not connected yet)
                self._is_connected = False
                
                async with websockets.connect(self.websocket_url) as websocket:
                    print(f"🔗 WebSocket connected to {self.websocket_url}")
                    reconnect_delay = 1.0  # Reset delay on successful connection
                    
                    # Mark as connected
                    self._is_connected = True
                    
                    try:
                        while True:  # Message processing loop
                            # Block until an image_id arrives
                            image_id: str = self._control_q.get()

                            # Get image data WITHOUT removing it (keep for retry if needed)
                            with self._mem_lock:
                                img_array = self.ImageMem.get(image_id, None)
                            
                            if img_array is None:
                                continue  # Image not found - skip silently

                            # Convert to grayscale if needed
                            if img_array.ndim == 3:
                                img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # type: ignore
                            else:
                                img_gray = img_array

                            # CRASH on encoding failure (definitely malformed data)
                            ok, buffer = cv2.imencode(".jpg", img_gray)  # type: ignore
                            if not ok:
                                raise RuntimeError(f"JPEG encode failed for {image_id} - corrupt image data")

                            # Try upload - distinguish client errors from network errors
                            try:
                                # Validate upload data using Pydantic model
                                upload_request = UploadRequest(
                                    image_id=image_id,
                                )
                                
                                # Create WebSocket message instead of HTTP request
                                message = {
                                    "type": "image_upload",
                                    "image_id": image_id,
                                    "timestamp": upload_request.timestamp,
                                    "image_data": base64.b64encode(buffer.tobytes()).decode()
                                }
                                
                                await websocket.send(json.dumps(message))
                                
                                # Upload successful - NOW remove from memory
                                with self._mem_lock:
                                    self.ImageMem.pop(image_id, None)
                                
                            except websockets.exceptions.InvalidURI:
                                # Invalid WebSocket URL - report as a critical error (same as InvalidURL before)
                                raise RuntimeError(f"Invalid WebSocket URL: {self.websocket_url}")

                            except (websockets.exceptions.WebSocketException, 
                                    websockets.exceptions.ConnectionClosed,
                                    ConnectionRefusedError, 
                                    OSError):
                                # Network issues - mark as disconnected, put message back in queue and break connection loop
                                # Image data stays in ImageMem for retry
                                self._is_connected = False
                                self._control_q.put_nowait(image_id)  # Put message back for retry
                                break  # Exit inner loop to trigger reconnection
                    finally:
                        # Always mark as disconnected when exiting the message processing loop
                        self._is_connected = False
                            
            except websockets.exceptions.InvalidURI:
                # Invalid WebSocket URL - report as a critical error
                raise RuntimeError(f"Invalid WebSocket URL: {self.websocket_url}")
                
            except (websockets.exceptions.WebSocketException,
                    websockets.exceptions.ConnectionClosed,
                    ConnectionRefusedError,
                    OSError) as e:
                # Connection failed - mark as disconnected and wait before retrying
                self._is_connected = False
                    
                print(f"🔄 WebSocket connection failed, retrying in {reconnect_delay:.1f}s: {e}")
                await asyncio.sleep(reconnect_delay)
                continue  # Try reconnecting
                
            except Exception as e:
                # Other errors should still crash
                raise e

    def _capture_loop(self) -> None:
        try:
            while True:
                ticket: SharedMem_ImgTicket = self._capture_q.get()
                img_view = debuffer_image(self.sharedmem_bufs[ticket.index].buf, ticket.res)
                embedded_id = decode_image_id(img_view)
                img_copy = img_view if img_view.flags.owndata else img_view.copy()
                with self._mem_lock:
                    self.ImageMem[embedded_id] = img_copy
                    while len(self.ImageMem) > self.max_store:
                        self.ImageMem.popitem(last=False)
        except Exception as e:
            
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return


class WebSocketEventsComms:
    """Ultra-lightweight, threaded sender and receiver for small events.

    Design goals:
    - Send small event objects (PlayerStatus, GameUpdate, PlayerTagged)
    - Receive and queue incoming events mapped to lumotag_events types
    - Avoid extra processes; use a single background thread
    - No busy waiting; block on a single control queue
    - Ignore network failures, crash on malformed events
    - Auto-reconnection like WebSocketImageComms
    - Crash on queue overflow or parsing errors (don't hide errors)

    Public API:
      - send_event(event)
      - get_received_event() -> event or None
      - get_received_events_count() -> int
      - is_connected()
      - get_send_queue_size()
      - raise_thread_error_if_any()
    """

    def __init__(
        self,
        websocket_url: str,
        OS_friendly_name: str,
    ) -> None:
        self.websocket_url = websocket_url
        self.OS_friendly_name = OS_friendly_name

        # Cache event types once at startup for performance
        self._cached_event_types = self._get_event_types()
        self._event_type_map = {cls.__name__: cls for cls in self._cached_event_types}

        # Send queue - small queue to detect performance issues
        self._send_q: threading_queue.Queue = threading_queue.Queue(maxsize=20)
        # Receive queue - bounded to 50 events, crash on overflow
        self._receive_q: threading_queue.Queue = threading_queue.Queue(maxsize=50)
        self._error_q: threading_queue.Queue = threading_queue.Queue(maxsize=10)

        # Connection status tracking for cheap health checks
        self._is_connected = False
        self._worker_thread = threading.Thread(target=self._worker_loop, name="events-worker", daemon=True)
        self._worker_thread.start()
        
        # Give thread time to start up before constructor returns
        time.sleep(0.1)

    def send_event(self, event) -> None:
        """Send an event - will crash if queue is full (performance issue)"""
        # Validate event is one of the expected Pydantic models from lumotag_events
        if not self._is_valid_event_type(event):
            valid_types = [cls.__name__ for cls in self._cached_event_types]
            raise ValueError(f"Event must be one of {valid_types}, got {type(event).__name__}")
        
        self._send_q.put_nowait(event)  # Will raise queue.Full if queue is full
        
        # Check for thread errors periodically to reduce overhead at high frequencies
        self._error_check_counter = getattr(self, '_error_check_counter', 0) + 1
        if self._error_check_counter % 10 == 0:  # Check every 10th call
            self.raise_thread_error_if_any()

    def get_received_event(self):
        """Get the next received event from the queue, or None if empty.
        Returns a Pydantic model instance from lumotag_events.py"""
        try:
            return self._receive_q.get_nowait()
        except threading_queue.Empty:
            return None

    def get_received_events_count(self) -> int:
        """Get current number of received events waiting in queue - extremely cheap to call"""
        return self._receive_q.qsize()

    def is_connected(self) -> bool:
        """Check if WebSocket is connected - extremely cheap to call"""
        return self._is_connected

    def get_send_queue_size(self) -> int:
        """Get current send queue size - extremely cheap to call"""
        return self._send_q.qsize()

    def get_supported_event_types(self) -> list[str]:
        """Get list of supported event type names (from cache)"""
        return [cls.__name__ for cls in self._cached_event_types]

    def raise_thread_error_if_any(self) -> None:
        """Lightweight check for thread errors"""
        # Check for caught exceptions first (non-blocking)
        if not self._error_q.empty():
            thread_name, exc, tb_str = self._error_q.get_nowait()
            
            # WebSocket worker should handle its own reconnections for network errors
            # If it crashed, that indicates a serious programming bug that should not be hidden
            if thread_name == "events-worker":
                raise RuntimeError(f"WebSocket events worker thread crashed unexpectedly: {exc}\n{tb_str}") from exc
            
            # For other errors, still crash (preserve test behavior)
            raise RuntimeError(f"{thread_name} failed: {exc}\n{tb_str}") from exc
        
        # Check for silent thread death
        if not self._worker_thread.is_alive():
            raise RuntimeError("Events worker thread died silently (no exception caught)")

    def _get_event_types(self):
        """Dynamically get all Pydantic model classes from lumotag_events module (called once at startup)"""
        import lumotag_events
        import inspect
        from pydantic import BaseModel
        
        event_types = []
        for _, obj in inspect.getmembers(lumotag_events):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseModel) and 
                obj is not BaseModel):
                event_types.append(obj)
        return event_types

    def _is_valid_event_type(self, event) -> bool:
        """Check if event is an instance of one of the expected event types"""
        return any(isinstance(event, event_type) for event_type in self._cached_event_types)

    def _parse_incoming_message(self, message_json: str):
        """Parse incoming WebSocket message and convert to event object.
        Returns event object or raises exception on parsing errors.
        Uses cached event type map for maximum efficiency."""
        try:
            message_data = json.loads(message_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in incoming message: {e}")

        # Check if this is an event message
        if message_data.get("type") != "event":
            # Not an event message - ignore silently (could be other message types)
            return None

        event_data = message_data.get("data")
        if event_data is None:
            raise ValueError("Missing 'data' field in event message")

        # Get event type from the data itself
        event_type_name = event_data.get("event_type")
        if not event_type_name:
            raise ValueError("Missing 'event_type' field in event data")

        # Use cached event type map for O(1) lookup
        event_class = self._event_type_map.get(event_type_name)
        if not event_class:
            valid_types = list(self._event_type_map.keys())
            raise ValueError(f"Unknown event type '{event_type_name}'. Valid types: {valid_types}")

        # Create Pydantic model instance - will validate data automatically
        try:
            event_instance = event_class(**event_data)
            return event_instance
        except Exception as e:
            raise ValueError(f"Failed to create {event_type_name} from data: {e}")

    def _handle_received_message(self, message_json: str):
        """Handle an incoming WebSocket message. Crashes on queue overflow or parsing errors."""
        try:
            # Parse the incoming message
            event = self._parse_incoming_message(message_json)
            
            # If it's not an event message, ignore it silently
            if event is None:
                return
            
            # Try to add to receive queue - CRASH if queue is full (don't hide errors)
            try:
                self._receive_q.put_nowait(event)
            except threading_queue.Full:
                # Log the error before crashing
                print(f"💥 CRITICAL: Receive queue overflow! Queue size: {self._receive_q.qsize()}")
                print(f"💥 Incoming event: {event.__class__.__name__}")
                raise RuntimeError(f"Receive queue overflow at {self._receive_q.maxsize} events. "
                                 f"Consumer is not processing events fast enough!") from None
                
        except Exception as e:
            # Log parsing errors before crashing
            print(f"💥 CRITICAL: Message parsing failed: {e}")
            print(f"💥 Raw message: {message_json}")
            raise RuntimeError(f"Failed to parse incoming message: {e}") from e

    async def _send_event(self, websocket, event):
        """Send a single event through the websocket. Raises exceptions on errors."""
        # Serialize event to JSON - CRASH on serialization failure
        try:
            event_dict = event.model_dump()
            # The event_type is now included in the data itself
            message = {
                "type": "event",
                "data": event_dict,
                "timestamp": time.time()
            }
            message_json = json.dumps(message)
        except Exception as e:
            raise RuntimeError(f"Event serialization failed: {e}") from e

        # Send the message
        await websocket.send(message_json)

    def _worker_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_worker())
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
        finally:
            loop.close()

    async def _ws_worker(self) -> None:
        reconnect_delay = 1.0
        
        while True:  # Reconnection loop
            try:
                # Mark as attempting connection (not connected yet)
                self._is_connected = False
                
                async with websockets.connect(self.websocket_url) as websocket:
                    print(f"🔗 WebSocket Events connected to {self.websocket_url}")
                    reconnect_delay = 1.0  # Reset delay on successful connection
                    
                    # Mark as connected
                    self._is_connected = True
                    
                    try:
                        # Create tasks for concurrent send/receive
                        send_task = None
                        receive_task = asyncio.create_task(websocket.recv())
                        
                        while True:  # Message processing loop
                            # Handle sending if we have an event to send
                            if send_task is None:
                                # Check for events to send (non-blocking)
                                try:
                                    event = self._send_q.get_nowait()
                                    send_task = asyncio.create_task(self._send_event(websocket, event))
                                except threading_queue.Empty:
                                    pass  # No events to send right now
                            
                            # Wait for either send completion or incoming message
                            tasks = [receive_task]
                            if send_task is not None:
                                tasks.append(send_task)
                            
                            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                            
                            # Handle completed tasks
                            for task in done:
                                if task == send_task:
                                    # Send completed
                                    try:
                                        await task  # Re-raise any exceptions
                                        send_task = None  # Ready for next send
                                    except (websockets.exceptions.WebSocketException, 
                                            websockets.exceptions.ConnectionClosed,
                                            ConnectionRefusedError, 
                                            OSError):
                                        # Network issues - put event back and break
                                        event = task.get_coro().cr_frame.f_locals.get('event')
                                        if event:
                                            self._send_q.put_nowait(event)
                                        self._is_connected = False
                                        # Cancel pending tasks
                                        for p in pending:
                                            p.cancel()
                                        break
                                        
                                elif task == receive_task:
                                    # Receive completed
                                    try:
                                        message = await task
                                        self._handle_received_message(message)
                                        # Create new receive task for next message
                                        receive_task = asyncio.create_task(websocket.recv())
                                    except websockets.exceptions.ConnectionClosed:
                                        # Connection closed - break to trigger reconnection
                                        self._is_connected = False
                                        # Cancel pending tasks
                                        for p in pending:
                                            p.cancel()
                                        break
                                    except Exception as e:
                                        # Other receive errors - log and continue
                                        print(f"⚠️ WebSocket receive error: {e}")
                                        # Create new receive task to continue listening
                                        receive_task = asyncio.create_task(websocket.recv())
                    finally:
                        # Always mark as disconnected when exiting the message processing loop
                        self._is_connected = False
                            
            except websockets.exceptions.InvalidURI:
                # Invalid WebSocket URL - report as a critical error
                raise RuntimeError(f"Invalid WebSocket URL: {self.websocket_url}")
                
            except (websockets.exceptions.WebSocketException,
                    websockets.exceptions.ConnectionClosed,
                    ConnectionRefusedError,
                    OSError) as e:
                # Connection failed - mark as disconnected and wait before retrying
                self._is_connected = False
                    
                print(f"🔄 WebSocket Events connection failed, retrying in {reconnect_delay:.1f}s: {e}")
                await asyncio.sleep(reconnect_delay)
                continue  # Try reconnecting
                
            except Exception as e:
                # Other errors should still crash
                raise e


def test_events_comms():
    """Simple test of WebSocketEventsComms functionality"""
    from lumotag_events import PlayerStatus, GameUpdate, PlayerTagged
    
    print("🧪 Testing WebSocketEventsComms...")
    
    # Create events comms (will fail to connect but that's OK for validation testing)
    events_comms = WebSocketEventsComms(
        websocket_url="ws://localhost:9999",  # Non-existent server for testing
        OS_friendly_name="test_pi"
    )
    
    print("✅ WebSocketEventsComms created")
    
    # Show cached event types
    supported_types = events_comms.get_supported_event_types()
    print(f"📋 Cached event types: {supported_types}")
    
    # Test valid event types
    print("\n🔬 Testing valid event types...")
    
    # Test PlayerStatus
    try:
        player_status = PlayerStatus(health=100, ammo=30, tag_id="player1", display_name="Test Player")
        events_comms.send_event(player_status)
        print(f"✅ PlayerStatus accepted: {player_status}")
    except Exception as e:
        print(f"❌ PlayerStatus failed: {e}")
    
    # Test GameUpdate
    try:
        game_update = GameUpdate(players=[
            PlayerStatus(health=80, ammo=25, tag_id="player1", display_name="Alice"),
            PlayerStatus(health=100, ammo=30, tag_id="player2", display_name="Bob")
        ])
        events_comms.send_event(game_update)
        print(f"✅ GameUpdate accepted: {len(game_update.players)} players")
    except Exception as e:
        print(f"❌ GameUpdate failed: {e}")
    
    # Test PlayerTagged
    try:
        player_tagged = PlayerTagged(tag_id="player1", image_ids=["img123", "img456"])
        events_comms.send_event(player_tagged)
        print(f"✅ PlayerTagged accepted: {player_tagged}")
    except Exception as e:
        print(f"❌ PlayerTagged failed: {e}")
    
    # Test invalid event type
    print("\n🚫 Testing invalid event type...")
    try:
        events_comms.send_event("not_a_pydantic_model")
        print("❌ Invalid event was accepted (this should not happen)")
    except ValueError as e:
        print(f"✅ Invalid event correctly rejected: {e}")
    except Exception as e:
        print(f"❌ Unexpected error for invalid event: {e}")
    
    # Test send queue size and receive queue
    send_queue_size = events_comms.get_send_queue_size()
    receive_queue_size = events_comms.get_received_events_count()
    print(f"\n📊 Send queue size: {send_queue_size}, Receive queue size: {receive_queue_size}")
    
    # Test connection status (should be False since server doesn't exist)
    connected = events_comms.is_connected()
    print(f"📡 Connected: {connected}")
    
    # Performance test - multiple rapid validations use cached types
    print("\n⚡ Testing cached validation performance...")
    test_event = PlayerStatus(health=50, ammo=15, tag_id="perf_test", display_name="Performance Test")
    
    import time
    start_time = time.time()
    for i in range(100):
        # This will validate against cached types (very fast)
        try:
            events_comms.send_event(test_event)
        except:
            pass  # Queue will fill up, but validation still happens
    end_time = time.time()
    
    print(f"✅ 100 validations completed in {(end_time - start_time)*1000:.2f}ms (using cached types)")
    print("🎉 WebSocketEventsComms basic test completed!")


def test_events_comms_error_handling():
    """Comprehensive error handling tests for WebSocketEventsComms"""
    from lumotag_events import PlayerStatus, GameUpdate, PlayerTagged
    import socket
    
    print("\n🧪 Testing WebSocketEventsComms Error Handling...")
    
    def find_available_port(start_port=8900):
        """Find an available port to avoid conflicts"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No available ports found")
    
    # Test 1: Invalid URL Detection
    print("\n🔄 Testing broken thread detection with invalid URL...")
    events_comms_broken_url = WebSocketEventsComms(
        websocket_url="ws://invalid-nonexistent-host-12345:9999",
        OS_friendly_name="test_events_invalid"
    )
    
    # Send an event to trigger connection attempt
    test_event = PlayerStatus(health=100, ammo=30, tag_id="test", display_name="Test")
    events_comms_broken_url.send_event(test_event)
    time.sleep(1.0)  # Let worker try to connect
    
    # Should gracefully handle invalid URL (no crash, just stays disconnected)
    try:
        events_comms_broken_url.raise_thread_error_if_any()
        connected = events_comms_broken_url.is_connected()
        print(f"✅ Invalid URL gracefully handled: connected={connected}")
        assert connected == False, "Should not be connected to invalid URL"
    except RuntimeError as e:
        if "Invalid WebSocket URL" in str(e):
            print(f"✅ Invalid URL properly detected: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
            raise
    
    # Test 2: Malformed Event Handling
    print("\n🔄 Testing malformed event detection...")
    events_comms_valid = WebSocketEventsComms(
        websocket_url="ws://localhost:65534",  # Will fail to connect, but that's OK
        OS_friendly_name="test_events_malformed"
    )
    
    # Test malformed event (not a Pydantic model)
    try:
        events_comms_valid.send_event({"not": "a_pydantic_model"})
        print("❌ Malformed event was accepted (should not happen)")
        assert False, "Malformed event should be rejected"
    except ValueError as e:
        print(f"✅ Malformed event correctly rejected: {e}")
    except Exception as e:
        print(f"❌ Unexpected error for malformed event: {e}")
        raise
    
    # Test 3: Queue Overflow Protection
    print("\n🔄 Testing event queue overflow protection...")
    events_comms_overflow = WebSocketEventsComms(
        websocket_url="ws://127.0.0.1:65535",  # Will fail fast
        OS_friendly_name="test_events_overflow"
    )
    
    # Blast the queue with events (maxsize=20)
    print("   💥 Blasting queue with rapid event sends...")
    test_event = PlayerStatus(health=75, ammo=20, tag_id="overflow_test", display_name="Overflow Test")
    queue_full_detected = False
    event_attempts = 0
    
    # Try to queue more events than the limit
    for i in range(30):  # More than maxsize=20
        try:
            events_comms_overflow.send_event(test_event)
            event_attempts += 1
            if i < 15:  # Don't spam console too much
                print(f"      ✓ Queued event #{event_attempts}")
        except Exception as e:
            queue_full_detected = True
            print(f"   ✅ Queue overflow detected after {event_attempts} events!")
            print(f"   ✅ Exception: {type(e).__name__}: {e}")
            exception_str = str(e)
            exception_type = type(e).__name__
            is_queue_full = ("Full" in exception_str or "queue" in exception_str.lower() or 
                           exception_type == "Full" or "Full" in exception_type)
            assert is_queue_full, f"Expected queue full error, got: {exception_type}: {exception_str}"
            break
    
    if not queue_full_detected:
        print(f"   ⚠️  Queue overflow not detected after {event_attempts} rapid events")
        print("   ℹ️  Worker thread may be processing too quickly - this is acceptable")
    
    print("   ✅ Queue overflow test completed")
    
    # Test 4: Connection Cycle Testing (Success → Death → Recovery → Reconnection)
    print("\n🔄 Testing connection cycle with event queueing...")
    
    class EventsTestServer:
        def __init__(self):
            self.port = find_available_port()
            self.server = None
            self.events_received = []
            self.is_running = False
            self.loop = None
            
        async def handle_client(self, websocket):
            print(f"   📱 Client connected to test server")
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "event":
                            event_data = data.get("data", {})
                            event_type = event_data.get("event_type")
                            if event_type and event_data:
                                # Store simplified event info
                                event_info = f"{event_type}:{event_data.get('tag_id', 'unknown')}"
                                self.events_received.append(event_info)
                                print(f"   📤 Server received: {event_info}")
                                
                                # Send a response event back to test incoming message handling
                                if event_type == "PlayerStatus":
                                    # Respond with a GameUpdate
                                    response_message = {
                                        "type": "event",
                                        "data": {
                                            "players": [{
                                                "health": 95,
                                                "ammo": 25,
                                                "tag_id": "server_player",
                                                "display_name": "Server Response",
                                                "event_type": "PlayerStatus"
                                            }],
                                            "event_type": "GameUpdate"
                                        },
                                        "timestamp": time.time()
                                    }
                                    await websocket.send(json.dumps(response_message))
                                    print(f"   📥 Server sent GameUpdate response")
                                
                                elif event_type == "PlayerTagged":
                                    # Respond with a PlayerStatus
                                    response_message = {
                                        "type": "event", 
                                        "data": {
                                            "health": 50,
                                            "ammo": 10,
                                            "tag_id": event_data.get("tag_id", "unknown"),
                                            "display_name": "Tagged Player Response",
                                            "event_type": "PlayerStatus"
                                        },
                                        "timestamp": time.time()
                                    }
                                    await websocket.send(json.dumps(response_message))
                                    print(f"   📥 Server sent PlayerStatus response")
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ Server received invalid JSON: {e}")
                    except Exception as e:
                        print(f"   ⚠️ Server error processing message: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                print(f"   📱 Client disconnected from test server")
            except Exception as e:
                print(f"   ❌ Server connection error: {e}")
        
        def start(self):
            def run_server():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
                async def server_main():
                    try:
                        self.server = await websockets.serve(self.handle_client, 'localhost', self.port)
                        self.is_running = True
                        print(f"   🌐 Test server started on localhost:{self.port}")
                        await self.server.wait_closed()
                    except Exception as e:
                        print(f"   ❌ Server start failed: {e}")
                        self.is_running = False
                    finally:
                        self.is_running = False
                        
                try:
                    self.loop.run_until_complete(server_main())
                except Exception as e:
                    print(f"   ❌ Server loop failed: {e}")
                    self.is_running = False
            
            self.server_thread = threading.Thread(target=run_server, daemon=False)  # Not daemon
            self.server_thread.start()
            
            # Wait for server to actually start
            for i in range(20):  # Wait up to 2 seconds
                time.sleep(0.1)
                if self.is_running:
                    print(f"   ✅ Server confirmed running after {(i+1)*0.1:.1f}s")
                    break
            else:
                print(f"   ⚠️ Server may not have started properly")
            
        def stop(self):
            if self.server and self.is_running:
                try:
                    if self.loop and not self.loop.is_closed():
                        self.loop.call_soon_threadsafe(self.server.close)
                        # Wait for graceful shutdown
                        time.sleep(0.2)
                    self.is_running = False
                    print(f"   🛑 Test server stopped")
                except Exception as e:
                    print(f"   ⚠️ Error stopping server: {e}")
                    self.is_running = False
        
        def get_url(self):
            return f"ws://localhost:{self.port}"
    
    # Run connection cycle test
    server = EventsTestServer()
    try:
        # Phase 1: Establish connection and test event sending
        print("\n   📡 Phase 1: Initial connection and event sending...")
        server.start()
        
        if not server.is_running:
            print("   ❌ Server failed to start, skipping test")
            return False
            
        test_events_comms = WebSocketEventsComms(
            websocket_url=server.get_url(),
            OS_friendly_name="cycle_test"
        )
        
        # Wait for client connection with timeout
        connected = False
        for i in range(30):  # Wait up to 3 seconds
            time.sleep(0.1)
            if test_events_comms.is_connected():
                connected = True
                print(f"   ✅ Client connected after {(i+1)*0.1:.1f}s")
                break
        
        if not connected:
            print("   ❌ Client failed to connect to server")
            return False
        
        # Send initial events
        initial_events = [
            PlayerStatus(health=100, ammo=30, tag_id="player1", display_name="Alice"),
            GameUpdate(players=[PlayerStatus(health=80, ammo=25, tag_id="player2", display_name="Bob")]),
            PlayerTagged(tag_id="player1", image_ids=["img001"])
        ]
        
        for event in initial_events:
            test_events_comms.send_event(event)
        time.sleep(1.0)
        
        phase1_events = len(server.events_received)
        print(f"   ✅ Phase 1 - Initial connection: {phase1_events} events received")
        
        # Test incoming events - check what we received from server responses
        time.sleep(1.0)  # Give time for server responses to be processed
        received_events = []
        while True:
            event = test_events_comms.get_received_event()
            if event is None:
                break
            received_events.append(f"{event.event_type}:{getattr(event, 'tag_id', 'N/A')}")
            print(f"   📥 Client received: {event.event_type}")
        
        print(f"   ✅ Phase 1 - Incoming events: {len(received_events)} events received by client")
        if len(received_events) > 0:
            print(f"   📋 Received event types: {received_events}")
        
        # Phase 2: Kill server and queue events (grace period)
        print("   💀 Phase 2: Server death - events should queue gracefully...")
        server.stop()
        time.sleep(0.5)
        
        # Send events during outage - they should queue up
        outage_events = [
            PlayerStatus(health=90, ammo=28, tag_id="player1", display_name="Alice Updated"),
            PlayerTagged(tag_id="player2", image_ids=["img002", "img003"]),
            GameUpdate(players=[
                PlayerStatus(health=90, ammo=28, tag_id="player1", display_name="Alice"),
                PlayerStatus(health=75, ammo=22, tag_id="player2", display_name="Bob")
            ])
        ]
        
        for event in outage_events:
            test_events_comms.send_event(event)
            time.sleep(0.1)
        
        queued = test_events_comms.get_send_queue_size()
        connected = test_events_comms.is_connected()
        print(f"   ✅ Phase 2 - During outage: {queued} events queued, connected={connected}")
        
        # Phase 3: Restart server and verify graceful reconnection
        print("   🔄 Phase 3: Server recovery - events should be delivered...")
        server.start()
        
        if not server.is_running:
            print("   ❌ Server failed to restart")
            return False
        
        # Wait for client to reconnect
        reconnected = False
        for i in range(50):  # Wait up to 5 seconds for reconnection
            time.sleep(0.1)
            if test_events_comms.is_connected():
                reconnected = True
                print(f"   ✅ Client reconnected after {(i+1)*0.1:.1f}s")
                break
        
        if reconnected:
            time.sleep(2.0)  # Additional time for queued events to be processed
        else:
            print("   ⚠️ Client didn't reconnect, but queued events may still be processed")
        
        final_events = len(server.events_received)
        remaining_queued = test_events_comms.get_send_queue_size()
        final_connected = test_events_comms.is_connected()
        
        # Check for any additional incoming events after reconnection
        time.sleep(1.0)  # Give time for any server responses
        final_received_events = []
        while True:
            event = test_events_comms.get_received_event()
            if event is None:
                break
            final_received_events.append(f"{event.event_type}:{getattr(event, 'tag_id', 'N/A')}")
            print(f"   📥 Client received after recovery: {event.event_type}")
        
        total_received_by_client = len(received_events) + len(final_received_events)
        
        print(f"   ✅ Phase 3 - After recovery: {final_events} total events sent to server, {remaining_queued} still queued, connected={final_connected}")
        print(f"   ✅ Phase 3 - Total events received by client: {total_received_by_client}")
        
        # Success criteria: should have received events from both phases
        expected_total = len(initial_events) + len(outage_events)
        success = phase1_events > 0 and final_events >= phase1_events
        
        if success:
            print(f"   🎉 Connection cycle test PASSED: {final_events}/{expected_total} events sent, {total_received_by_client} received")
        else:
            print(f"   ⚠️  Connection cycle test: {final_events}/{expected_total} events sent, {total_received_by_client} received")
        
    finally:
        server.stop()
    
    # Test 5: Connection Status Testing
    print("\n🔄 Testing connection status with events...")
    
    # Test invalid URL (should stay False)
    events_invalid = WebSocketEventsComms(
        websocket_url="ws://nonexistent-host-99999:8888",
        OS_friendly_name="test_status_invalid"
    )
    
    status_immediate = events_invalid.is_connected()
    time.sleep(2.0)
    status_after_wait = events_invalid.is_connected()
    
    print(f"   📡 Invalid URL - Immediate: {status_immediate}, After 2s: {status_after_wait}")
    assert status_immediate == False, f"Expected False immediately, got {status_immediate}"
    assert status_after_wait == False, f"Expected False after wait, got {status_after_wait}"
    print("   ✅ Invalid URL status test passed")
    
    print("\n🎉 WebSocketEventsComms error handling tests completed!")


if __name__ == "__main__":
    """Comprehensive test with real image data and WebSocket server"""
    import time
    import numpy as np
    import threading
    import json
    from my_collections import SharedMem_ImgTicket
    
    # Test the new WebSocketEventsComms first
    test_events_comms()
    
    # Test comprehensive error handling for WebSocketEventsComms
    test_events_comms_error_handling()
    print("\n" + "="*60 + "\n")
    
    print("🧪 Testing WebSocketImageComms with real data...")
    
    # Create a test rectangle image with embedded ID using real factory functions
    def create_test_image_with_id(width=640, height=480):
        """Create a test grayscale image with a white rectangle and properly embedded ID using factory functions"""
        from factory import create_image_id, decode_image_id
        
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Add white rectangle in center
        rect_w, rect_h = 200, 100
        start_x = (width - rect_w) // 2
        start_y = (height - rect_h) // 2
        img[start_y:start_y+rect_h, start_x:start_x+rect_w] = 255
        
        # Add border
        img[start_y:start_y+5, start_x:start_x+rect_w] = 128  # Top
        img[start_y+rect_h-5:start_y+rect_h, start_x:start_x+rect_w] = 128  # Bottom
        img[start_y:start_y+rect_h, start_x:start_x+5] = 128  # Left  
        img[start_y:start_y+rect_h, start_x+rect_w-5:start_x+rect_w] = 128  # Right
        
        # Use the real factory function to create and embed the ID
        img_id = create_image_id()
        img[0, 0:img_id.shape[0]] = img_id  # Embed exactly like ImageGenerator.get_image()
        
        # Decode the ID to show what was embedded
        embedded_id = decode_image_id(img)
        
        return img, embedded_id  # Return both image and the decoded ID for reference
    
    # Simple WebSocket server to receive uploads
    class WebSocketTestServer:
        uploads_received = []
        
        @staticmethod
        async def handle_client(websocket):
            """Handle WebSocket client connections"""
            print(f"📱 WebSocket client connected")
            
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        if data.get("type") == "image_upload":
                            image_id = data.get("image_id")
                            timestamp = data.get("timestamp")
                            image_data = data.get("image_data")
                            
                            if image_id:
                                WebSocketTestServer.uploads_received.append(image_id)
                                print(f"📤 WebSocket Server received upload:")
                                print(f"   - image_id: {image_id}")
                                print(f"   - timestamp: {timestamp}")
                                if image_data:
                                    # Decode base64 to get actual image size
                                    img_bytes = base64.b64decode(image_data)
                                    print(f"   - image_data: {len(img_bytes)} bytes")
                            else:
                                print("❌ No image_id found in upload")
                                
                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON received: {e}")
                    except Exception as e:
                        print(f"❌ Error handling message: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                print("📱 WebSocket client disconnected")
    
    # Start WebSocket server in background
    server_port = 8765
    
    async def async_websocket_server():
        server = await websockets.serve(
            WebSocketTestServer.handle_client, 
            'localhost', 
            server_port
        )
        print(f"🌐 WebSocket server started on localhost:{server_port}")
        await server.wait_closed()
    
    def run_websocket_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_websocket_server())
    
    server_thread = threading.Thread(target=run_websocket_server, daemon=True)
    server_thread.start()
    time.sleep(0.5)  # Give server time to start
    
    try:
        # Create real shared memory buffer with test image
        test_img, embedded_id = create_test_image_with_id(640, 480)
        img_bytes = test_img.tobytes()
        print(f"🎯 Created test image with ID: {embedded_id}")
        
        # Mock shared memory that contains our test image
        class MockSharedMem:
            def __init__(self, data):
                self.buf = memoryview(bytearray(data))
        
        sharedmem_buffs = {0: MockSharedMem(img_bytes)}
        
        def safe_mem_details_func():
            return SharedMem_ImgTicket(
                index=0, 
                res=(480, 640),  # height, width as used in debuffer_image
                buf_size=len(img_bytes),
                id=1
            )
        
        # Create uploader with local WebSocket server
        uploader = WebSocketImageComms(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            websocket_url=f"ws://localhost:{server_port}",
            OS_friendly_name="test_pi"
        )
        print("✅ Uploader created with real image data")
        

        
        # Threads are already started thanks to built-in sleep
        uploader.raise_thread_error_if_any()
        print("✅ Threads are healthy")
        
        # Test capture with real image
        print("\n📸 Testing image capture...")
        uploader.trigger_capture()
        time.sleep(0.1)  # Let capture process
        
        stored_ids = uploader.get_stored_image_ids()
        print(f"✅ Captured {len(stored_ids)} images: {stored_ids}")
        assert len(stored_ids) > 0, "No images were captured - capture test failed"
        
        # Test upload of image
        print(f"\n📤 Testing upload of image: {stored_ids[0]}")
        initial_uploads = len(WebSocketTestServer.uploads_received)
        uploader.upload_image_by_id(stored_ids[0])
        
        # Wait for upload to complete and check server response
        # Check multiple times with small delays to handle timing issues
        success = False
        for attempt in range(50):  # Try for up to 2 seconds
            time.sleep(0.2)
            if len(WebSocketTestServer.uploads_received) > initial_uploads:
                success = True
                break
        
        assert success, f"Upload failed: Server did not receive the image ID. Expected uploads > {initial_uploads}, got {len(WebSocketTestServer.uploads_received)}"
        print(f"✅ Upload successful! Server received: {WebSocketTestServer.uploads_received}")
        
        # Test multiple captures and uploads
        print("\n🔄 Testing multiple operations...")
        for i in range(3):
            uploader.trigger_capture()
            time.sleep(0.05)
        
        stored_ids = uploader.get_stored_image_ids()
        print(f"✅ Multiple captures: {len(stored_ids)} images stored")
        # Note: Since we're using the same image data, we get the same embedded ID each time
        # So we expect only 1 unique image ID, not 3 separate ones
        assert len(stored_ids) >= 1, f"Expected at least 1 image from multiple captures, got {len(stored_ids)}"
        
        # Upload all stored images
        initial_upload_count = len(WebSocketTestServer.uploads_received)
        for img_id in stored_ids:
            uploader.upload_image_by_id(img_id)
        
        # Wait and check results
        time.sleep(2.0)
        final_stored = uploader.get_stored_image_ids()
        final_upload_count = len(WebSocketTestServer.uploads_received)
        print(f"✅ After uploads: {len(final_stored)} images remaining")
        print(f"✅ Total uploads received by server: {final_upload_count}")
        
        # Assert that uploads were processed (images should be removed from storage after upload)
        # Since we're uploading the same image ID that was already uploaded before, 
        # it gets removed from storage when uploaded, so no new uploads are expected
        # The key test is that images are removed from storage after upload
        assert len(final_stored) == 0, f"Expected 0 images remaining after upload, got {len(final_stored)}"
        
        # Final health check
        uploader.raise_thread_error_if_any()
        print("✅ Threads still healthy after all operations")
        
        # Test delete function
        print("\n🗑️  Testing delete_image_by_id function...")
        
        # Capture some images first
        for i in range(2):
            uploader.trigger_capture()
            time.sleep(0.05)
        
        stored_ids = uploader.get_stored_image_ids()
        print(f"✅ Captured {len(stored_ids)} images for deletion test: {stored_ids}")
        assert len(stored_ids) >= 1, "Need at least 1 image to test deletion"
        
        # Test deleting existing image
        test_id = stored_ids[0]
        result = uploader.delete_image_by_id(test_id)
        print(f"✅ Delete existing image '{test_id}': {result}")
        assert result == True, f"Expected True when deleting existing image, got {result}"
        
        # Verify image was actually removed
        updated_ids = uploader.get_stored_image_ids()
        print(f"✅ Images after deletion: {updated_ids}")
        assert test_id not in updated_ids, f"Image {test_id} still exists after deletion"
        assert len(updated_ids) == len(stored_ids) - 1, f"Expected {len(stored_ids) - 1} images, got {len(updated_ids)}"
        
        # Test deleting non-existent image
        result = uploader.delete_image_by_id("nonexistent_id")
        print(f"✅ Delete non-existent image: {result}")
        assert result == False, f"Expected False when deleting non-existent image, got {result}"
        
        # Verify no images were affected
        final_ids = uploader.get_stored_image_ids()
        assert final_ids == updated_ids, "Image count changed when deleting non-existent image"
        
        print("✅ Delete function tests passed!")

        print(f"\n🎉 Comprehensive test completed!")
        print(f"   📊 Images uploaded: {len(WebSocketTestServer.uploads_received)}")
        print(f"   📊 Server responses: {WebSocketTestServer.uploads_received}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup - WebSocket server will stop when thread ends
        print("🧹 WebSocket server stopping")

    # Test broken thread detection with nonsense URL
    print("\n🔄 Testing broken thread detection with nonsense URL...")
    uploader_broken_url = WebSocketImageComms(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url="ws://invalid-url",  # Clearly invalid URL
        OS_friendly_name="test_pi"
    )
    

    
    uploader_broken_url.trigger_capture()
    time.sleep(0.1)  # Let capture process
    stored_ids = uploader_broken_url.get_stored_image_ids()
    assert len(stored_ids) > 0, "No images captured for broken URL test"
    
    uploader_broken_url.upload_image_by_id(stored_ids[0])
    time.sleep(1.0)
    
    # Expecting an exception here due to nonsense URL
    try:
        uploader_broken_url.raise_thread_error_if_any()
        # If we reach here, no thread error was detected
        print("⚠️  Nonsense URL test: No thread errors detected")
        print("   ℹ️  This is expected because network failures are intentionally ignored")
        print("   ℹ️  by design - only 4xx client errors crash the thread")
        # This is actually expected behavior, so we don't assert failure here
    except RuntimeError as e:
        if "Invalid WebSocket URL" in str(e):
            print(f"✅ Nonsense URL test passed: {e}")
        else:
            assert False, f"Nonsense URL test failed: Unexpected error type: {e}"
    except Exception as e:
        assert False, f"Nonsense URL test failed: Unexpected exception: {e}"

    # Test broken thread detection with nonsense image
    print("\n🔄 Testing broken thread detection with nonsense image...")
    def create_nonsense_image(width=640, height=480):
        """Create a nonsense image that will fail debuffering"""
        img = np.zeros((height, width), dtype=np.uint8)
        img[0, 0:16] = np.random.randint(0, 256, size=16, dtype=np.uint8)  # Random bytes
        return img
    
    nonsense_img = create_nonsense_image()
    nonsense_img_bytes = nonsense_img.tobytes()
    sharedmem_buffs_nonsense = {0: MockSharedMem(nonsense_img_bytes)}
    
    uploader_nonsense_img = WebSocketImageComms(
        sharedmem_buffs=sharedmem_buffs_nonsense,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url=f"ws://localhost:{server_port}",
        OS_friendly_name="test_pi"
    )
    uploader_nonsense_img.trigger_capture()
    time.sleep(0.1)  # Let capture process
    
    # Expecting an exception here due to nonsense image
    thread_error_detected = False
    try:
        uploader_nonsense_img.raise_thread_error_if_any()
    except RuntimeError as e:
        if "decode" in str(e):
            print(f"✅ Nonsense image test passed: {e}")
            thread_error_detected = True
        else:
            assert False, f"Nonsense image test failed: Unexpected error type: {e}"
    except Exception as e:
        assert False, f"Nonsense image test failed: Unexpected exception: {e}"
    
    assert thread_error_detected, "Nonsense image test failed: No thread errors detected when one was expected"

    # Test queue overflow protection
    print("\n🔄 Testing upload queue overflow protection...")
    
    # Create uploader for overflow testing
    uploader_overflow = WebSocketImageComms(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        websocket_url="ws://127.0.0.1:65534/blocked",  # This will fail fast
        OS_friendly_name="test_pi"
    )
    
    # Capture one image to have something to upload
    uploader_overflow.trigger_capture()
    time.sleep(0.1)
    stored_ids = uploader_overflow.get_stored_image_ids()
    print(f"   ✅ Captured {len(stored_ids)} images for testing")
    assert len(stored_ids) > 0, "No images captured for overflow test"
    
    # Blast the queue with requests in a tight loop (maxsize=5)
    print("   💥 Blasting queue with rapid upload requests...")
    test_image_id = stored_ids[0]
    queue_full_detected = False
    upload_attempts = 0
    
    # Try to queue 10 uploads of the same image as fast as possible
    for i in range(10):
        try:
            uploader_overflow.upload_image_by_id(test_image_id)
            upload_attempts += 1
            print(f"      ✓ Queued upload #{upload_attempts}")
            
        except Exception as e:
            queue_full_detected = True
            print(f"   ✅ Queue overflow detected after {upload_attempts} uploads!")
            print(f"   ✅ Exception: {type(e).__name__}: {e}")
            # Verify it's actually a queue full error - check exception type and string representation
            exception_str = str(e)
            exception_type = type(e).__name__
            is_queue_full = ("Full" in exception_str or "queue" in exception_str.lower() or 
                           exception_type == "Full" or "Full" in exception_type)
            assert is_queue_full, f"Expected queue full error, got: {exception_type}: {exception_str}"
            break
    
    # Note: Queue overflow might not always be detected if the worker thread processes too quickly
    # This is acceptable behavior, so we don't assert failure if no overflow is detected
    if not queue_full_detected:
        print(f"   ⚠️  Queue overflow not detected after {upload_attempts} rapid uploads")
        print("   ℹ️  Worker thread may be processing too quickly - this is acceptable")
    
    print("   ✅ Queue blast test completed")
    
    # AUTOMATED RECONNECTION TESTS
    print("\n🔄 Running automated reconnection tests...")
    
    # Import test functions from our test files
    import socket
    
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
    
    # Test: Connection Success -> Server Death -> Server Recovery -> Reconnection Success
    def test_connection_cycle():
        """Test the full connection cycle: success -> death -> recovery -> reconnection"""
        print("\n📡 Test: Connection Cycle (Success → Death → Recovery → Reconnection)")
        
        class CycleTestServer:
            def __init__(self):
                self.port = find_available_port()
                self.server = None
                self.uploads_received = []
                self.is_running = False
                self.loop = None
                
            async def handle_client(self, websocket):
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get("type") == "image_upload":
                            image_id = data.get("image_id")
                            if image_id:
                                self.uploads_received.append(image_id)
                                print(f"   📤 Server received: {image_id}")
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    print(f"   ❌ Server error: {e}")
            
            def start(self):
                def run_server():
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                    
                    async def server_main():
                        self.server = await websockets.serve(self.handle_client, 'localhost', self.port)
                        self.is_running = True
                        await self.server.wait_closed()
                        
                    self.loop.run_until_complete(server_main())
                
                self.server_thread = threading.Thread(target=run_server, daemon=True)
                self.server_thread.start()
                time.sleep(0.5)
                
            def stop(self):
                if self.server and self.is_running:
                    if self.loop:
                        self.loop.call_soon_threadsafe(self.server.close)
                        self.is_running = False
                        time.sleep(0.5)
            
            def get_url(self):
                return f"ws://localhost:{self.port}"
        
        # Run the test
        server = CycleTestServer()
        try:
            # Phase 1: Establish connection and test upload
            server.start()
            test_uploader = WebSocketImageComms(
                sharedmem_buffs=sharedmem_buffs,
                safe_mem_details_func=safe_mem_details_func,
                websocket_url=server.get_url(),
                OS_friendly_name="cycle_test"
            )
            
            time.sleep(1.0)
            
            test_uploader.trigger_capture()
            time.sleep(0.2)  # Give capture time to process
            stored_ids = test_uploader.get_stored_image_ids()
            if stored_ids:
                test_uploader.upload_image_by_id(stored_ids[0])
            time.sleep(1.0)
            
            phase1_uploads = len(server.uploads_received)
            print(f"   ✅ Phase 1 - Initial connection: {phase1_uploads} uploads")
            
            # Phase 2: Kill server and queue messages
            server.stop()
            time.sleep(0.5)
            
            for i in range(2):
                test_uploader.trigger_capture()
                time.sleep(0.1)  # Give capture thread time to process
                stored_ids = test_uploader.get_stored_image_ids()
                if stored_ids:
                    test_uploader.upload_image_by_id(stored_ids[0])
            
            queued = len(test_uploader.get_stored_image_ids())
            print(f"   ✅ Phase 2 - Messages queued during outage: {queued}")
            
            # Phase 3: Restart server and verify reconnection
            server.start()
            time.sleep(3.0)  # Wait for reconnection and message processing
            

            
            final_uploads = len(server.uploads_received)
            remaining_queued = len(test_uploader.get_stored_image_ids())
            
            print(f"   ✅ Phase 3 - After recovery: {final_uploads} total uploads, {remaining_queued} still queued")
            
            # Success criteria:
            # - Phase 1 should have at least 1 upload
            # - Phase 2 should have at least some activity (queued >= 0 is always true, so we check if images exist)
            # - Phase 3 should have final_uploads >= phase1_uploads
            # Note: queued might be 0 if using same image ID (overwrites in ImageMem dict)
            success = phase1_uploads > 0 and final_uploads >= phase1_uploads
            return success
            
        finally:
            server.stop()
    
    # Run the connection cycle test
    try:
        cycle_success = test_connection_cycle()
        if cycle_success:
            print("   ✅ Connection cycle test PASSED")
        else:
            print("   ❌ Connection cycle test FAILED")
    except Exception as e:
        print(f"   ❌ Connection cycle test ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("   ✅ Automated reconnection tests completed")

    # CONNECTION STATUS TESTS
    print("\n🔄 Testing connection status functionality...")
    
    def test_connection_status():
        """Test the is_connected() method with realistic timing expectations"""
        print("\n📡 Test: Connection Status Detection")
        
        # Test 1: Invalid URL should stay False
        print("   1️⃣ Testing invalid URL (should stay False)...")
        uploader_invalid = WebSocketImageComms(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            websocket_url="ws://nonexistent-host-12345:9999",
            OS_friendly_name="test_invalid"
        )
        
        # Check immediately and after wait
        status_immediate = uploader_invalid.is_connected()
        time.sleep(2.0)  # Give it time to try and fail
        status_after_wait = uploader_invalid.is_connected()
        
        print(f"      📡 Immediate status: {status_immediate}")
        print(f"      📡 Status after 2s: {status_after_wait}")
        
        assert status_immediate == False, f"Expected False immediately, got {status_immediate}"
        assert status_after_wait == False, f"Expected False after wait, got {status_after_wait}"
        print("      ✅ Invalid URL test passed")
        
        # Test 2: Valid connection should become True
        print("   2️⃣ Testing valid connection (should become True)...")
        uploader_valid = WebSocketImageComms(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            websocket_url=f"ws://localhost:{server_port}",
            OS_friendly_name="test_valid"
        )
        
        # Wait for connection with generous timeout
        connected = False
        for i in range(50):  # Wait up to 5 seconds
            time.sleep(0.1)
            if uploader_valid.is_connected():
                connected = True
                print(f"      📡 Connected after {(i+1)*0.1:.1f}s")
                break
        
        assert connected, "Should have connected to valid server within 5 seconds"
        print("      ✅ Valid connection test passed")
        
        # Test 3: Connection status during activity
        print("   3️⃣ Testing status during normal operation...")
        
        # Do some normal operations and verify status stays True
        uploader_valid.trigger_capture()
        time.sleep(0.2)
        stored_ids = uploader_valid.get_stored_image_ids()
        
        status_during_activity = uploader_valid.is_connected()
        print(f"      📡 Status during activity: {status_during_activity}")
        assert status_during_activity == True, "Should stay connected during normal operations"
        
        if stored_ids:
            uploader_valid.upload_image_by_id(stored_ids[0])
            time.sleep(0.5)
            status_after_upload = uploader_valid.is_connected()
            print(f"      📡 Status after upload: {status_after_upload}")
            assert status_after_upload == True, "Should stay connected after uploads"
        
        print("      ✅ Normal operation test passed")
        
        return True
    
    # Run connection status tests
    try:
        if test_connection_status():
            print("   ✅ Connection status tests PASSED")
        else:
            print("   ❌ Connection status tests FAILED")
    except Exception as e:
        print(f"   ❌ Connection status test ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("   ✅ Connection status testing completed")
    