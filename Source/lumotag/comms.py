import time
import cv2
import requests
from multiprocessing import Process, Queue
import threading
import queue as threading_queue
from typing import Callable
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import traceback
from analyse_lumotag import debuffer_image
from factory import decode_image_id
from my_collections import SharedMem_ImgTicket


# UploadResult removed - we no longer track upload results
# Network failures are expected and ignored, encoding failures crash immediately


class ImageUploaderThreaded_shared_mem:
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
      - get_stored_image_ids()
      - raise_thread_error_if_any()
    """

    def __init__(
        self,
        sharedmem_buffs: dict,
        safe_mem_details_func: Callable[[], SharedMem_ImgTicket],
        upload_url: str,
        OS_friendly_name: str,
        max_store: int = 20,
    ) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.safe_mem_details_func = safe_mem_details_func
        self.upload_url = upload_url
        self.OS_friendly_name = OS_friendly_name
        self.max_store = max_store

        # Keep raw grayscale frames by embedded image id
        self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_lock = threading.Lock()

        # Separate queues to decouple capture (debuffer+copy) and upload (encode+HTTP)
        # Small queues like ImageAnalyser - should be empty if processing fast enough
        self._capture_q: threading_queue.Queue = threading_queue.Queue(maxsize=1)
        # Upload control: image_id strings only - small queue to detect performance issues
        self._control_q: threading_queue.Queue = threading_queue.Queue(maxsize=5)
        # upload_result_q removed - no longer tracking upload results
        self._error_q: threading_queue.Queue = threading_queue.Queue(maxsize=10)

        self._capture_thread = threading.Thread(target=self._capture_loop, name="uploader-capture", daemon=True)
        self._upload_thread = threading.Thread(target=self._worker_loop, name="uploader-worker", daemon=True)
        self._capture_thread.start()
        self._upload_thread.start()

    def trigger_capture(self) -> None:
        """Trigger capture - will crash if queue is full (performance issue)"""
        ticket = self.safe_mem_details_func()
        self._capture_q.put_nowait(ticket)  # Will raise queue.Full if queue is full
        
        # Check for thread errors after every trigger for performance characterization
        # TODO: Consider pseudo-random checking (e.g., 1 in 50 calls) for production to reduce overhead
        self.raise_thread_error_if_any()

    def upload_image_by_id(self, image_id: str) -> None:
        """Queue a specific image ID for upload - will crash if queue is full"""
        self._control_q.put_nowait(image_id)

    def get_stored_image_ids(self) -> list[str]:
        with self._mem_lock:
            return list(self.ImageMem.keys())

    # get_upload_result() removed - no longer tracking upload results

    def raise_thread_error_if_any(self) -> None:
        """Check for thread errors and silent thread death"""
        # Check for caught exceptions first
        if not self._error_q.empty():
            thread_name, exc, tb_str = self._error_q.get_nowait()
            raise RuntimeError(f"{thread_name} failed: {exc}\n{tb_str}") from exc
        
        # Check for silent thread death (threads died without raising exceptions)
        if not self._capture_thread.is_alive():
            raise RuntimeError("Capture thread died silently (no exception caught)")
        if not self._upload_thread.is_alive():
            raise RuntimeError("Upload thread died silently (no exception caught)")

    def _worker_loop(self) -> None:
        session = requests.Session()
        try:
            while True:
                # Block until an image_id arrives
                image_id: str = self._control_q.get()

                # Single lock acquisition - get and remove image atomically
                with self._mem_lock:
                    img_array = self.ImageMem.pop(image_id, None)
                
                if img_array is None:
                    continue  # Image not found - skip silently

                # Convert to grayscale if needed
                if img_array.ndim == 3:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
                else:
                    img_gray = img_array

                # CRASH on encoding failure (definitely malformed data)
                ok, buffer = cv2.imencode(".jpg", img_gray)  # type: ignore[attr-defined]
                if not ok:
                    raise RuntimeError(f"JPEG encode failed for {image_id} - corrupt image data")

                # Try upload - distinguish client errors from network errors
                try:
                    files = {"image": (f"{image_id}.jpg", buffer.tobytes(), "image/jpeg")}
                    data = {
                        "image_id": image_id,
                        "timestamp": time.time(),
                        "source": self.OS_friendly_name,
                        "mono": 1,
                    }
                    resp = session.post(self.upload_url, files=files, data=data, timeout=2)
                    resp.raise_for_status()
                    # Upload successful - no result tracking needed
                    
                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and 400 <= e.response.status_code < 500:
                        # 4xx = malformed request, CRASH
                        raise RuntimeError(f"Malformed upload request for {image_id}: HTTP {e.response.status_code}")
                    # 5xx = server error, ignore (network/server issue)
                    
                except (requests.exceptions.RequestException, 
                        requests.exceptions.Timeout, 
                        requests.exceptions.ConnectionError):
                    # Network issues - ignore and continue
                    pass
        except Exception as e:
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return

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

