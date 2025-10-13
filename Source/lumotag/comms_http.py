import time
import cv2
import base64
import json
import threading
import queue as threading_queue
from typing import Callable
from collections import OrderedDict, deque
import numpy as np
import traceback
import requests
import logging
from functools import lru_cache, wraps
from analyse_lumotag import debuffer_image
from factory import decode_image_id
from my_collections import SharedMem_ImgTicket
from utils import time_it
import lumotag_events
import inspect
from pydantic import BaseModel
from abc import ABC, abstractmethod
from dataclasses import dataclass
import socket
import ipaddress
import random

TAGDAM = 25 # damage when any player gets tagged


class ImageDecodeError(Exception):
    """Raised when image decoding fails"""
    pass

class DisplayName(str):
    """Type-safe device identifier"""
    pass

class DeviceID(str):
    """Type-safe device identifier"""
    pass

class TagID(str):
    """Type-safe device identifier"""
    pass

@dataclass
class EventWithCallback:
    """Wrapper for events that need a callback function executed after successful transmission"""
    event: BaseModel
    callback: Callable[[dict], None] | None

@dataclass
class LogEvent:
    """Event log entry with text and log level"""
    text: str
    level: int  # Use logging.INFO, logging.WARNING, logging.CRITICAL


def _get_broadcast_address() -> str:
    """Calculate broadcast address from local IP (assumes /24 subnet)"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Doesn't actually send data
        local_ip = s.getsockname()[0]
        s.close()
        
        # Assume /24 subnet (most common)
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        return str(network.broadcast_address)
    except Exception:
        # Fallback to limited broadcast
        return "255.255.255.255"


class AbstractHTTPComms(ABC):
    @abstractmethod
    def __init__(
        self,
        sharedmem_buffs_closerange: dict,
        safe_mem_details_func_closerange: Callable[[], SharedMem_ImgTicket],
        sharedmem_buffs_longrange: dict,
        safe_mem_details_func_longrange: Callable[[], SharedMem_ImgTicket],
        images_url: str,
        events_url: str,
        gamestate_url: str,
        avatar_files_url: str,
        OS_friendly_name: str,
        device_id: str,
        upload_timeout: float = 0.5,
        poll_interval_seconds: float = 0.3,
    ) -> None:
        pass
    
    @abstractmethod
    def trigger_capture_close_range(self) -> None:
        pass

    @abstractmethod
    def trigger_capture_long_range(self) -> None:
        pass

    @abstractmethod
    def get_upload_queue_size(self) -> int:
        pass
    
    @abstractmethod
    def upload_image_by_id(self, image_id: str) -> None:
        pass
    
    @abstractmethod
    def delete_image_by_id(self, image_id: str) -> bool:
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def send_tagging_event(self, tag_id: str, image_ids: list[str]) -> None:
        pass

    @abstractmethod
    def get_player_avatar(self, tag_id: str) -> np.ndarray:
        pass


class HTTPComms(AbstractHTTPComms):
    """Ultra-lightweight, threaded HTTP uploader for grayscale frames from shared memory.

    Design goals:
    - Use HTTP POST requests instead of WebSocket
    - Four separate threads: capture, image upload, event sending, gamestate polling
    - No busy waiting; block on queues
    - Keep a bounded cache of most recent frames (raw grayscale)
    - Encode and POST only on explicit request
    - Ignore network failures, crash on malformed requests
    - Validate events using lumotag_events types

    Public API:
      - trigger_capture_close_range()
      - upload_image_by_id(image_id)
      - delete_image_by_id(image_id) 
      - send_event(event)
      - raise_thread_error_if_any()
    """

    def __init__(
        self,
        sharedmem_buffs_closerange: dict,
        safe_mem_details_func_closerange: Callable[[], SharedMem_ImgTicket],
        sharedmem_buffs_longrange: dict,
        safe_mem_details_func_longrange: Callable[[], SharedMem_ImgTicket],
        images_url: str,
        events_url: str,
        gamestate_url: str,
        avatar_files_url: str,
        OS_friendly_name: str,
        device_id: str,
        killshots_of_me: list[np.ndarray] = [],
        upload_timeout: float = 0.5,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.sharedmem_buffs_closerange = sharedmem_buffs_closerange
        self.safe_mem_details_func_closerange = safe_mem_details_func_closerange
        self.sharedmem_buffs_longrange = sharedmem_buffs_longrange
        self.safe_mem_details_func_longrange = safe_mem_details_func_longrange
        self.images_url = images_url.rstrip('/')  # Remove trailing slash
        self.events_url = events_url.rstrip('/')  # Remove trailing slash
        self.gamestate_url = gamestate_url.rstrip('/')  # Remove trailing slash
        self.avatar_files_url = avatar_files_url.rstrip('/')  # Remove trailing slash
        self.OS_friendly_name = OS_friendly_name
        self.device_id = device_id
        self.upload_timeout = upload_timeout
        self.max_cached_images = 20  # Maximum number of images to keep in memory before dropping oldest - in theory should just need last 2 frames
        self.poll_interval_seconds = poll_interval_seconds

        # UDP broadcast setup (fire-and-forget, ~0.1ms overhead)
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._udp_sock.setblocking(False)  # Guarantee sendto() never blocks
        self._udp_broadcast_addr = _get_broadcast_address()
        self._udp_port = 5000
        print(f"UDP broadcast configured: {self._udp_broadcast_addr}:{self._udp_port}")
        
        # UDP listener socket (separate socket for receiving)
        self._udp_listener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_listener_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_listener_sock.bind(('', self._udp_port))  # Listen on all interfaces

        # Keep raw grayscale frames by embedded image id
        self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_lock = threading.Lock()
        
        # Connection state tracking (thread-safe)
        self._is_connected = True
        self._connection_lock = threading.Lock()
        self._last_success_time = time.time()
        self._last_events_error_time = None  # Initialize to avoid linter error
        self._last_disconnect_log_time = 0.0  # Rate limit disconnect messages
        self._last_error_check_time = 0.0  # Rate limit error checking (once per second)
        
        # UDP / HTTP tag notification (thread-safe sticky flag)
        self._udp_tagged = False
        self._GameUpdate_tagged = False
        self._tagged_lock = threading.Lock()
        

        # Game state tracking (thread-safe)
        self._latest_gamestate =  lumotag_events.GameStatus(players={})
        self._gamestate_lock = threading.Lock()
        self._player_avatars: dict[DeviceID, np.ndarray] = {}
        self._tag_id_to_display_name: dict[TagID, str] = {}
        
        # Event log queue (FIFO with max 50 items, lock-free thread-safe)
        # deque append() and popleft() are atomic from opposite ends
        self.events_queue = deque(maxlen=50)  # Automatically drops oldest when full

        # Cache event types once at startup for performance (from WebSocketEventsComms)
        self._cached_event_types = self._get_event_types()
        self._event_type_map = {cls.__name__: cls for cls in self._cached_event_types}
        
        # Separate queues for different concerns
        self._capture_q_close_range: threading_queue.Queue = threading_queue.Queue(maxsize=2)
        self._capture_q_long_range: threading_queue.Queue = threading_queue.Queue(maxsize=2)
        self._upload_q: threading_queue.Queue = threading_queue.Queue(maxsize=3)
        self._events_q: threading_queue.Queue = threading_queue.Queue(maxsize=5)
        self._error_q: threading_queue.Queue = threading_queue.Queue(maxsize=5)


        
        # Error checking counter
        self._error_check_counter = 0
        
        # Worker threads
        self._capture_thread_closerange = threading.Thread(target=self._capture_loop, args=(self._capture_q_close_range,self.sharedmem_buffs_closerange,), name="http-capture-close-range", daemon=True)
        self._capture_thread_longrange = threading.Thread(target=self._capture_loop, args=(self._capture_q_long_range,self.sharedmem_buffs_longrange,), name="http-capture-long-range", daemon=True)
        self._upload_thread = threading.Thread(target=self._upload_worker, name="http-upload", daemon=True)
        self._events_thread = threading.Thread(target=self._events_worker, name="http-events", daemon=True)
        self._gamestate_thread = threading.Thread(target=self._gamestate_worker, name="http-gamestate", daemon=True)
        self._udp_listener_thread = threading.Thread(target=self._udp_listener_worker, name="udp-listener", daemon=True)
        
        # Start all threads
        self._capture_thread_longrange.start()
        self._capture_thread_closerange.start()
        self._upload_thread.start()
        self._events_thread.start()
        self._gamestate_thread.start()
        self._udp_listener_thread.start()
        
        # Give threads time to start up before constructor returns
        time.sleep(0.1)

    def trigger_capture_close_range(self) -> None:
        """Trigger capture - will crash if queue is full (performance issue)"""
        ticket = self.safe_mem_details_func_closerange()
        self._capture_q_close_range.put_nowait(ticket)  # Will raise queue.Full if queue is full

    def trigger_capture_long_range(self) -> None:
        """Trigger capture - will crash if queue is full (performance issue)"""
        ticket = self.safe_mem_details_func_longrange()
        self._capture_q_long_range.put_nowait(ticket)  # Will raise queue.Full if queue is full

    def upload_image_by_id(self, image_id: str) -> None:
        """Queue a specific image ID for upload from memory"""
        upload_task = lumotag_events.UploadFromMemoryRequest(image_id=image_id)
        try:
            self._upload_q.put_nowait(upload_task)
        except threading_queue.Full:
            self.add_event_to_log("Upload queue full!", logging.WARNING)
    
    def upload_image_from_disk(self, file_path: str, image_id: str) -> None:
        """Queue a disk-based image upload"""
        upload_task = lumotag_events.UploadFromDiskRequest(
            image_id=image_id,
            file_path=file_path
        )
        try:
            self._upload_q.put_nowait(upload_task)
        except threading_queue.Full:
            self.add_event_to_log("Upload queue full!", logging.WARNING)

    def set_killshot(self, response: dict):
        try:
            incoming_killshot = lumotag_events.ReqKillScreenResponse(**response)
            
            # Decode each base64 image
            for i, image_data in enumerate(incoming_killshot.image_datas):
                # Skip any null/empty images (shouldn't happen but be defensive)
                if not image_data:
                    print(f"⚠️ Killshot image {i} is empty, skipping")
                    continue
                    
                # Decode base64 to bytes
                image_bytes = base64.b64decode(image_data)
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Decode JPEG to OpenCV image
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    self.killshots_of_me.append(image)
                else:
                    print(f"⚠️ Failed to decode killshot image {i}")
        except Exception as e:
            # Don't crash the thread if killshot processing fails
            print(f"⚠️ Failed to process killshot response: {e}")
            self.add_event_to_log("Killshot unavailable (may have been cleared)", logging.WARNING)

        # # Debug: Show images
        # for i, image in enumerate(decoded_images):
        #     cv2.imshow(f"Killshot {i+1} from {incoming_killshot.display_name_tagger}", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    def download_avatar(self, tagid: str, timeout: float = 1.0) -> np.ndarray:

        url = f"{self.avatar_files_url}/{tagid}.jpg"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code != 200:
            raise ImageDecodeError(f"Failed to download {url}: HTTP {response.status_code}")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(response.content, np.uint8)
        
        # Decode image to OpenCV format
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ImageDecodeError(f"Failed to decode image from {url}")
        
        return image

    def _broadcast_udp(self, event: lumotag_events.PlayerTagged) -> None:
        """Fire-and-forget UDP broadcast (~0.1ms, never blocks or crashes on network errors)
        
        Safe even with no network connectivity - kernel drops packet if no interface.
        Raises on serialization errors (programming bugs).
        """
        # Let serialization errors crash (programming bugs)
        udp_payload = event.model_dump_json().encode('utf-8')
        
        try:
            # Only catch network-related errors
            self._udp_sock.sendto(udp_payload, (self._udp_broadcast_addr, self._udp_port))
        except OSError:
            pass  # Network error (no interface, buffer full, etc) - silently ignore

    def request_kill_screen(self):
        self.killshots_of_me = []
        event = lumotag_events.ReqKillScreen()
        try:
            self._events_q.put_nowait(EventWithCallback(event, self.set_killshot))
        except threading_queue.Full:
            self.add_event_to_log("KILLSCRN: Events queue full!", logging.WARNING)

    def send_tagging_event(self, tag_ids: list[str], image_ids: list[str]) -> None:
        """Send a tagging event via HTTP and UDP broadcast - validates event type first"""
        
        event = lumotag_events.PlayerTagged(tag_ids=tag_ids, image_ids=image_ids)

        # UDP broadcast first for fastest peer notification
        self._broadcast_udp(event)

        # HTTP event queued for reliable delivery
        try:
            self._events_q.put_nowait(EventWithCallback(event, None))
        except threading_queue.Full:
            self.add_event_to_log("TAG: Events queue full!", logging.WARNING)

    def delete_image_by_id(self, image_id: str) -> bool:
        """Delete a specific image ID from storage - returns True if deleted, False if not found"""
        with self._mem_lock:
            return self.ImageMem.pop(image_id, None) is not None

    def get_upload_queue_size(self) -> int:
        """Get current upload queue size - extremely cheap to call"""
        return self._upload_q.qsize()

    def get_events_queue_size(self) -> int:
        """Get current events queue size - extremely cheap to call"""
        return self._events_q.qsize()

    def is_connected(self) -> bool:
        """Check if currently connected - thread-safe
        Also checks for thread errors (rate-limited internally to once per second)"""
        self.raise_thread_error_if_any()
        with self._connection_lock:
            return self._is_connected
    
    def acknowledge_tagEvent(self) -> bool:
        """Check if we've been tagged via UDP broadcast or HTTP update request, and clear the flag (acknowledge)
        
        Returns:
            True if we were tagged since last check, False otherwise
        """
        with self._tagged_lock:
            was_tagged = self._udp_tagged or self._GameUpdate_tagged
            self._udp_tagged = False
            self._GameUpdate_tagged = False
            return was_tagged
    
    def get_latest_gamestate(self) -> lumotag_events.GameStatus:
        """Get most recent game state (non-blocking, thread-safe)
        Returns validated GameStatus Pydantic object"""
        with self._gamestate_lock:
            return self._latest_gamestate
    
    def add_event_to_log(self, event_text: str, level: int = logging.INFO) -> None:
        """Add an event to the event log queue (lock-free thread-safe, FIFO with auto-eviction)
        
        Args:
            event_text: Message to log
            level: Log level (use logging.INFO, logging.WARNING, logging.CRITICAL)
        """
        self.events_queue.append(LogEvent(text=event_text, level=level))  # Atomic operation
    
    def pop_oldest_event(self) -> LogEvent | None:
        """Get and remove the oldest event from the queue (lock-free thread-safe)
        
        Returns:
            LogEvent with text and level, or None if queue is empty
        """
        try:
            return self.events_queue.popleft()  # Atomic operation
        except IndexError:
            return None  # Queue was empty

    def _set_connected(self, connected: bool, usermsg: str = None) -> None:
        """Internal method to update connection state - thread-safe"""
        if connected is False:
            # Rate limit disconnect messages to once every 2 seconds
            current_time = time.time()
            if current_time - self._last_disconnect_log_time >= 2.0:
                self.add_event_to_log(f"OFFLINE::{usermsg}", logging.WARNING)
                self._last_disconnect_log_time = current_time
        with self._connection_lock:
            if connected and not self._is_connected:
                # Reconnected - update success time
                self._last_success_time = time.time()
            self._is_connected = connected

    def raise_thread_error_if_any(self) -> None:
        """Lightweight check for thread errors (rate-limited to once per second)"""
        # Rate limit full checks to once per second
        current_time = time.time()
        if current_time - self._last_error_check_time < 1.0:
            return  # Skip check, too soon
        
        self._last_error_check_time = current_time
        
        # Check for caught exceptions first (non-blocking)
        if not self._error_q.empty():
            thread_name, exc, tb_str = self._error_q.get_nowait()
            raise RuntimeError(f"{thread_name} failed: {exc}\n{tb_str}") from exc
        
        # Check for silent thread death
        if not self._capture_thread_closerange.is_alive():
            raise RuntimeError("Capture thread died silently (no exception caught)")
        if not self._upload_thread.is_alive():
            raise RuntimeError("Upload worker thread died silently (no exception caught)")
        if not self._events_thread.is_alive():
            raise RuntimeError("Events worker thread died silently (no exception caught)")
        if not self._gamestate_thread.is_alive():
            raise RuntimeError("Gamestate worker thread died silently (no exception caught)")
        if not self._udp_listener_thread.is_alive():
            raise RuntimeError("UDP listener thread died silently (no exception caught)")

    def _get_event_types(self):
        """Dynamically get all Pydantic model classes from lumotag_events module (called once at startup)"""
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

    def _capture_loop(
            self,
            capture_q: threading_queue.Queue,
            sharedmem_bufs: dict,
            ) -> None:
        """Capture thread - handles image debuffering and copying"""
        try:
            while True:
                ticket: SharedMem_ImgTicket = capture_q.get()  # Blocks until work available
                img_view = debuffer_image(sharedmem_bufs[ticket.index].buf, ticket.res)
                embedded_id = decode_image_id(img_view)
                
                # Do expensive copy outside the lock to minimize lock contention
                img_copy = img_view.copy()
                
                with self._mem_lock:
                    # Store raw image data - encode to JPEG only when uploading (lazy encoding)
                    self.ImageMem[embedded_id] = img_copy
                    # Remove oldest images if cache is full (FIFO eviction)
                    while len(self.ImageMem) > self.max_cached_images:
                        _ , _ = self.ImageMem.popitem(last=False)
                        
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return

    def _upload_worker(self) -> None:
        """Upload worker thread - handles HTTP POST requests for images"""
        # Optimized session for fast image uploads
        session = requests.Session()
        session.headers.update({
            "x-device-id": self.device_id,
            "Content-Type": "application/json",
            "Connection": "keep-alive"  # Keep connections open
        })
        
        # Configure connection pooling for speed
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,    # Only need 1 connection to image server
            pool_maxsize=1,        # Keep 1 connection alive
            max_retries=0          # No retries - fail fast
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        try:
            while True:
                upload_item = self._upload_q.get()  # Blocks until work available
                
                # Check if this is a memory-based upload or disk-based upload
                if isinstance(upload_item, lumotag_events.UploadFromMemoryRequest):
                    # Memory-based upload (common case)
                    image_id = upload_item.image_id
                    
                    # Get image data WITHOUT removing it (keep for retry if needed)
                    with self._mem_lock:
                        img_data = self.ImageMem.get(image_id, None)
                    
                    if img_data is None:
                        continue  # Image not found - skip silently
                elif isinstance(upload_item, lumotag_events.UploadFromDiskRequest):
                    # Disk-based upload (rare case)
                    image_id = upload_item.image_id
                    img_data = cv2.imread(upload_item.file_path, cv2.IMREAD_COLOR)
                    if img_data is None:
                        raise RuntimeError(f"Failed to load image from disk: {upload_item.file_path}")
                else:
                    raise RuntimeError(f"Unexpected upload item type: {type(upload_item)} - expected UploadFromMemoryRequest or UploadFromDiskRequest")
                
                try:
                    # Image data must be raw numpy array - encode to JPEG for upload
                    if not isinstance(img_data, np.ndarray):
                        raise RuntimeError(f"Expected numpy array for image {image_id}, got {type(img_data)}")
                    
                    # Encode raw image to JPEG
                    ok, jpeg_buffer = cv2.imencode(".jpg", img_data)
                    if not ok:
                        raise RuntimeError(f"JPEG encode failed for {image_id} - corrupt image data")
                    
                    # Create complete upload request with image data
                    upload_request = lumotag_events.UploadRequest(
                        image_id=image_id,
                        image_data=base64.b64encode(jpeg_buffer.tobytes()).decode()
                    )
                    
                    # Clean POST payload - pure Pydantic model serialization
                    # All transport metadata is in headers
                    post_data = upload_request.model_dump()
                    

                    # Fast HTTP POST - headers already set in session
                    response = session.post(
                        self.images_url,
                        json=post_data,
                        timeout=self.upload_timeout
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        # Upload successful - remove from memory and mark as connected
                        with self._mem_lock:
                            self.ImageMem.pop(image_id, None)
                        self._set_connected(True, "imgupload ok")
                    else:
                        # Server error - keep image for retry and mark as disconnected for 4xx/5xx errors
                        if response.status_code >= 400:
                            self._set_connected(False, f"imgupload {response.status_code}")
                            time.sleep(0.5)  # Small delay to prevent tight retry loops
                        print(f"⚠️ Image upload failed: HTTP {response.status_code}")
                        
                except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                    # Network error or timeout - keep image for retry and mark as disconnected
                    err_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                    self._set_connected(False, f"imgupload {err_msg}")
                    time.sleep(0.5)  # Shorter delay for faster recovery
  
                    
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return

    def _events_worker(self) -> None:
        """Events worker thread - handles HTTP POST requests for events"""
        # Optimized session for fast event sending
        session = requests.Session()
        session.headers.update({
            "x-device-id": self.device_id,
            "Content-Type": "application/json",
            "Connection": "keep-alive"  # Keep connections open
        })
        
        # Configure connection pooling for speed
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,    # Only need 1 connection to events server
            pool_maxsize=1,        # Keep 1 connection alive
            max_retries=0          # No retries - fail fast
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        try:
            while True:
                event: EventWithCallback = self._events_q.get()  # Blocks until work available
                
                try:
                    # Clean POST payload - only Pydantic model data
                    # All transport metadata is in headers
                    post_data = event.event.model_dump()
                    
                    # Fast HTTP POST - headers already set in session
                    response = session.post(
                        self.events_url,
                        json=post_data,
                        timeout=0.2  # Fast timeout for events (200ms)
                    )
                    
                    # Check response
                    if response.status_code == 200:
                        self._set_connected(True, "event ok")  # Success - mark as connected
                        if event.callback is not None:
                            event.callback(response.json())
                    else:
                        # Server error - mark as disconnected for 4xx/5xx errors
                        if response.status_code >= 400:
                            # self._set_connected(False, f"event {response.status_code}")
                            
                            # Special handling for killscreen failures (404 = images deleted/unavailable)
                            if response.status_code == 404 and event.event.event_type == "ReqKillScreen":
                                self.add_event_to_log("Killshot unavailable (cleared by server)", logging.WARNING)
                            
                            time.sleep(0.5)  # Small delay to prevent tight retry loops
                        print(f"⚠️ Event send failed: HTTP {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    # Timeout - mark disconnected, no log spam for frequent events
                    self._set_connected(False, "event timeout")
                    
                except requests.exceptions.RequestException as e:
                    # Network error - mark as disconnected
                    err_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                    self._set_connected(False, f"event {err_msg}")
                    time.sleep(0.1)  # Shorter delay for faster recovery
                    # Don't print every network error to avoid spam
                    if self._last_events_error_time is None or time.time() - self._last_events_error_time > 10.0:
                        print(f"⚠️ Events network error: {e}")
                        self._last_events_error_time = time.time()
                    
                # except Exception as e:
                #     # Other errors - log and continue
                #     print(f"⚠️ Event send error: {e}")
                    
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return

    def _gamestate_worker(self) -> None:
        """Gamestate worker thread - polls server for game updates every poll_interval"""
        
        # Optimized session for fast polling
        session = requests.Session()
        session.headers.update({
            "x-device-id": self.device_id,
            "Connection": "keep-alive",  # Keep connections open
            "Cache-Control": "no-cache"   # Don't cache responses
        })
        
        # Configure connection pooling for speed
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,    # Only need 1 connection to gamestate server
            pool_maxsize=1,        # Keep 1 connection alive
            max_retries=0          # No retries - fail fast
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        try:
            while True:
                try:
                    # Fast HTTP GET with minimal timeout for LAN
                    response = session.get(
                        self.gamestate_url,
                        timeout=0.1  # 100ms timeout for LAN - fail fast
                    )
                    
                    # Check response - only accept 200 OK
                    if response.status_code == 200:
                        # Parse and validate response as GameStatus (Pydantic validation kept for strict data integrity)
                        try:
                            gamestate_data = response.json()
                        
                            game_update = lumotag_events.GameStatus(**gamestate_data)  # Keep Pydantic validation - it's fast and necessary
                            

                            # Check for player eliminations and taggings
                            for device_id, player in game_update.players.items():
                                old_player = self._latest_gamestate.players.get(device_id)
                                if old_player and not old_player.isEliminated and player.isEliminated:
                                    # self.add_event_to_log(f"{player.display_name} eliminated!")
                                    self.add_event_to_log(random.choice(lumotag_events.eliminated_chat).substitute(player_name=player.display_name), logging.CRITICAL)
                                elif old_player and player.health < old_player.health:
                                    self.add_player_tagged_to_log(player.display_name, player.health)


                            # Check if we got tagged (health decreased)
                            if (self._latest_gamestate.players 
                                and self.device_id in game_update.players 
                                and self.device_id in self._latest_gamestate.players 
                                and game_update.players[self.device_id].health < self._latest_gamestate.players[self.device_id].health):
                                damage = self._latest_gamestate.players[self.device_id].health - game_update.players[self.device_id].health
                                with self._tagged_lock:
                                    self._GameUpdate_tagged = True
                            


                            # Store validated game state
                            with self._gamestate_lock:
                                self._latest_gamestate = game_update
                            

                            # check here if we have everyones avatars
                            for deviceid, playerstatus in self._latest_gamestate.players.items():
                                if playerstatus.display_name not in self._player_avatars:
                                    self._player_avatars[DisplayName(playerstatus.display_name)] = self.download_avatar(playerstatus.display_name)
                                    self._tag_id_to_display_name[TagID(str(playerstatus.tag_id))] = playerstatus.display_name
                                    # Player joined - log it (unless it's us)
                                    if deviceid != self.device_id:
                                        self.add_event_to_log(f"{playerstatus.display_name} [{playerstatus.tag_id}] joined",logging.WARNING)
                                    break

                            self._set_connected(True, "gamestate ok")  # Success - mark as connected
                            
                        except Exception as e:
                            # JSON parsing or validation error - this is a serious issue
                            print(f"Invalid gamestate response format: {e}")
                            err_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                            self._set_connected(False, f"gamestate {err_msg}")
                            
                    else:
                        # Any non-200 response is an error - mark as disconnected
                        self._set_connected(False, f"gamestate {response.status_code}")
                        print(f"Gamestate fetch failed: HTTP {response.status_code}")
                        # Don't call raise_for_status() - it's slow and unnecessary
                        
                except requests.exceptions.Timeout:
                    # Timeout - mark as disconnected but don't spam logs
                    self._set_connected(False, "gamestate req timeout")
                    
                except requests.exceptions.RequestException as e:
                    # Network error - mark as disconnected
                    err_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                    self._set_connected(False, f"gamestate {err_msg}")
                    print(f"Gamestate network error: {e}")
                    
                except Exception as e:
                    # Unexpected error - reraise to be caught by outer handler
                    raise
                
                # Wait for next poll cycle
                time.sleep(self.poll_interval_seconds)

        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return

    def _udp_listener_worker(self) -> None:
        """UDP listener thread - blocks on recvfrom() (0% CPU when idle, ~0.2ms per packet)"""
        try:
            while True:
                # Blocks here until packet arrives (thread sleeps, 0% CPU)
                data, addr = self._udp_listener_sock.recvfrom(1024)
                
                try:
                    # Parse JSON payload
                    message = json.loads(data.decode('utf-8'))
                    
                    event_type = message.get('event_type')
                    
                    if event_type == 'PlayerTagged':
                        # Deserialize into Pydantic model for validation
                        event = lumotag_events.PlayerTagged(**message)
                        
                        # Check if it's us getting tagged - grab gamestate reference quickly
                        with self._gamestate_lock:
                            gamestate = self._latest_gamestate
                        
                        # Work with local reference outside lock (safe - Pydantic immutable)
                        if gamestate and gamestate.players:
                            my_player = gamestate.players.get(self.device_id)
                            if my_player and my_player.tag_id in event.tag_ids:
                                # Set sticky flag - main thread will check and clear it
                                with self._tagged_lock:
                                    self._udp_tagged = True
                                self.reduce_players_health(self.device_id, TAGDAM) 


                except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as e:
                    # Malformed packet - log and continue
                    print(f"UDP: Invalid packet from {addr}: {e}")
                    
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return
    
    def add_player_tagged_to_log(self, player_name, player_heath, device_id: str | None = None):
        """provide DeviceID if you can which will check if its ourselves getting damaged"""
        if player_heath < 0:
            message = random.choice(lumotag_events.bullied_chat).substitute(player_name=player_name, health=player_heath)
        else:
            message = random.choice(lumotag_events.tagged_chat).substitute(player_name=player_name)
        if device_id == self.device_id:
            loglvl = logging.CRITICAL
        else:
            loglvl = logging.INFO
        self.add_event_to_log(message, loglvl)


    def reduce_players_health(self, device_id: str, damage: int):
        """Reduce player health locally (note: gamestate polling will overwrite this)
        """
        try:
            if device_id in self._latest_gamestate.players:
                with self._gamestate_lock:
                    self._latest_gamestate.players[device_id].health -= damage
                self.add_player_tagged_to_log(
                    self._latest_gamestate.players[device_id].display_name,
                    self._latest_gamestate.players[device_id].health,
                    device_id = device_id
                    )

        except Exception as e:
            # If this fails, crash the whole process - something is seriously wrong
            tb_str = traceback.format_exc()
            self._error_q.put_nowait((threading.current_thread().name, e, tb_str))

    def get_player_avatar(self, display_name: DisplayName):
        # avatars are downloaded asynchronously when players join the server - so try again 
        return self._player_avatars.get(display_name, None)
    
    def get_display_name(self, tag_id: TagID) -> str | None:
        """Get display name for a tag_id, returns None if not found"""
        return self._tag_id_to_display_name.get(tag_id, None)