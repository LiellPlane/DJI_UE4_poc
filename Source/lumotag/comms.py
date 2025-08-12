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

from analyse_lumotag import debuffer_image
from factory import decode_image_id
from my_collections import SharedMem_ImgTicket


@dataclass
class UploadResult:
    """Result of an upload operation"""
    imageid: str
    success: bool
    error_message: str = None
    upload_time_ms: float = 0


class ImageUploader_shared_mem:
    """Class to handle image uploads using shared memory as input"""
    
    def __init__(
            self,
            sharedmem_buffs: dict,
            safe_mem_details_func: Callable[[], SharedMem_ImgTicket],
            upload_url: str,
            OS_friendly_name: str,
            max_queue_size: int = 100) -> None:
        
        self.sharedmem_bufs = sharedmem_buffs
        self.upload_url = upload_url
        self.OS_friendly_name = OS_friendly_name
        self.safe_mem_details_func = safe_mem_details_func
        self.input_shared_mem_index_q = Queue(maxsize=max_queue_size)
        self.upload_request_q = Queue(maxsize=50)  # Queue for specific upload requests
        self.upload_result_q = Queue(maxsize=1)
        self.last_capture_time = time.perf_counter()
        self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        
        func_args = (
            self.input_shared_mem_index_q,
            self.upload_request_q,
            self.upload_result_q)

        process = Process(
            target=self.async_upload_loop,
            args=func_args,
            daemon=True)

        process.start()

    def check_if_timed_out(self):
        """Check if capture process has timed out"""
        if time.perf_counter() - self.last_capture_time > 10:  # wait in seconds
            return True
        return False

    def get_capture_time_ms(self):
        """Get time since last capture in milliseconds"""
        return (time.perf_counter() - self.last_capture_time) * 1000

    def trigger_capture(self):
        """Trigger capture of current shared memory buffer to store in ImageMem"""
        if not self.input_shared_mem_index_q.full():  # skip if queue is full
            self.last_capture_time = time.perf_counter()  # reset timeout
            self.input_shared_mem_index_q.put(
                self.safe_mem_details_func(),
                block=False,
                timeout=None)

    def upload_image_by_id(self, image_id: str):
        """Queue a specific image ID for upload"""
        if not self.upload_request_q.full():
            self.upload_request_q.put(image_id, block=False)

    def get_stored_image_ids(self) -> list[str]:
        """Get list of currently stored image IDs"""
        return list(self.ImageMem.keys())

    def async_upload_loop(
            self,
            input_shared_mem_index_q,
            upload_request_q,
            upload_result_q):
        """Main loop running in separate process - handles both image storage and uploads"""
        
        while True:
            # Always block until a new image is available; main loop posts frequently
            shared_details = input_shared_mem_index_q.get(block=True, timeout=None)
            self._store_image_from_shared_mem(shared_details)
            
            # Check for upload requests (non-blocking)
            try:
                image_id = upload_request_q.get(block=False)
                self._handle_upload_request(image_id, upload_result_q)
            except:
                pass  # No upload requests
            
            # Clean up old images (same as analyse_lumotag.py)
            if len(self.ImageMem) > 100:
                _, _ = self.ImageMem.popitem(last=False)
            
            # No sleep needed; loop blocks on image arrival

    def _store_image_from_shared_mem(self, shared_details):
        """Store image from shared memory into ImageMem"""
        try:
            # Read image from shared memory (zero copy)
            img_buff = debuffer_image(
                self.sharedmem_bufs[shared_details.index].buf,
                shared_details.res)
            
            # Get embedded image ID
            embedded_id = decode_image_id(img_buff)
            
            # Store image (copy if needed, same as analyse_lumotag.py)
            if img_buff.flags.owndata:
                self.ImageMem[embedded_id] = img_buff
            else:
                self.ImageMem[embedded_id] = img_buff.copy()
                
        except Exception as e:
            print(f"Error storing image: {e}")

    def _handle_upload_request(self, image_id: str, upload_result_q):
        """Handle upload request for specific image ID"""
        start_time = time.perf_counter()
        
        try:
            if image_id in self.ImageMem:
                # Upload the image
                success = self._upload_image(image_id, self.ImageMem[image_id])
                
                # Remove from memory after upload attempt
                del self.ImageMem[image_id]
                
                upload_time_ms = (time.perf_counter() - start_time) * 1000
                result = UploadResult(
                    imageid=image_id,
                    success=success,
                    upload_time_ms=upload_time_ms)
            else:
                # Image not found
                result = UploadResult(
                    imageid=image_id,
                    success=False,
                    error_message="Image not found in memory",
                    upload_time_ms=0)
                
        except Exception as e:
            upload_time_ms = (time.perf_counter() - start_time) * 1000
            result = UploadResult(
                imageid=image_id,
                success=False,
                error_message=str(e),
                upload_time_ms=upload_time_ms)
        
        # Send result back (non-blocking)
        try:
            upload_result_q.put(result, block=False)
        except:
            pass  # Queue full, skip result

    def _upload_image(self, image_id: str, img_array) -> bool:
        """Upload image to server"""
        try:
            # Encode image as JPEG
            success, buffer = cv2.imencode('.jpg', img_array)  # type: ignore
            if not success:
                print(f"Failed to encode image {image_id}")
                return False
            
            # Prepare upload data
            files = {
                'image': (f'{image_id}.jpg', buffer.tobytes(), 'image/jpeg')
            }
            data = {
                'image_id': image_id,
                'timestamp': time.time(),
                'source': self.OS_friendly_name
            }
            
            # Upload with timeout
            response = requests.post(
                self.upload_url, 
                files=files, 
                data=data, 
                timeout=5)
            response.raise_for_status()
            
            print(f"Successfully uploaded {image_id} from {self.OS_friendly_name}")
            return True
            
        except Exception as e:
            print(f"Upload failed for {image_id}: {e}")
            return False

    def get_upload_result(self) -> UploadResult | None:
        """Get upload result if available (non-blocking)"""
        try:
            return self.upload_result_q.get(block=False)
        except:
            return None


class ImageUploaderThreaded_shared_mem:
    """Ultra-lightweight, threaded uploader for grayscale frames from shared memory.

    Design goals:
    - Avoid extra processes; use a single background thread
    - No busy waiting; block on a single control queue
    - Keep a bounded cache of most recent frames (raw grayscale)
    - Encode and POST only on explicit request

    Public API mirrors the process-based variant for drop-in usage:
      - trigger_capture()
      - upload_image_by_id(image_id)
      - get_stored_image_ids()
      - get_upload_result()
      - raise_thread_error_if_any()  # raise if a worker thread reported an error
    """

    def __init__(
        self,
        sharedmem_buffs: dict,
        safe_mem_details_func: Callable[[], SharedMem_ImgTicket],
        upload_url: str,
        OS_friendly_name: str,
        max_store: int = 50,
        throttle_interval_s: float = 1.0,
    ) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.safe_mem_details_func = safe_mem_details_func
        self.upload_url = upload_url
        self.OS_friendly_name = OS_friendly_name
        self.max_store = max_store
        self.throttle_interval_s = throttle_interval_s  # kept for API compatibility; not used

        # Keep raw grayscale frames by embedded image id
        self.ImageMem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_lock = threading.Lock()

        # Separate queues to decouple capture (debuffer+copy) and upload (encode+HTTP)
        self._capture_q: threading_queue.Queue = threading_queue.Queue(maxsize=10)
        # Upload control: image_id strings only
        self._control_q: threading_queue.Queue = threading_queue.Queue(maxsize=200)
        self.upload_result_q: threading_queue.Queue = threading_queue.Queue(maxsize=1)
        self._error_q: threading_queue.Queue = threading_queue.Queue(maxsize=10)

        self._capture_thread = threading.Thread(target=self._capture_loop, name="uploader-capture", daemon=True)
        self._upload_thread = threading.Thread(target=self._worker_loop, name="uploader-worker", daemon=True)
        self._capture_thread.start()
        self._upload_thread.start()

    def trigger_capture(self) -> None:
        # Enqueue ticket; capture thread will debuffer+copy immediately
        try:
            ticket = self.safe_mem_details_func()
            self._capture_q.put_nowait(ticket)
        except Exception:
            pass

    def upload_image_by_id(self, image_id: str) -> None:
        try:
            self._control_q.put_nowait(image_id)
        except Exception:
            pass

    def get_stored_image_ids(self) -> list[str]:
        with self._mem_lock:
            return list(self.ImageMem.keys())

    def get_upload_result(self) -> UploadResult | None:
        try:
            return self.upload_result_q.get_nowait()
        except threading_queue.Empty:
            return None

    def raise_thread_error_if_any(self) -> None:
        """Raise the first worker-thread exception if present (non-blocking)."""
        try:
            thread_name, exc, tb_str = self._error_q.get_nowait()
        except threading_queue.Empty:
            return None
        raise RuntimeError(f"{thread_name} failed: {exc}\n{tb_str}") from exc

    def _worker_loop(self) -> None:
        session = requests.Session()
        try:
            while True:
                # Block until an image_id arrives
                image_id: str = self._control_q.get()
                start_ts = time.perf_counter()

                with self._mem_lock:
                    present = image_id in self.ImageMem
                if not present:
                    res = UploadResult(
                        imageid=image_id,
                        success=False,
                        error_message="Image not found in memory",
                        upload_time_ms=0,
                    )
                    try:
                        self.upload_result_q.put_nowait(res)
                    except Exception:
                        pass
                    continue

                with self._mem_lock:
                    img_array = self.ImageMem.get(image_id)
                if img_array is None:
                    raise RuntimeError("Image disappeared from cache before upload")

                if img_array.ndim == 3:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # type: ignore[attr-defined]
                else:
                    img_gray = img_array

                ok, buffer = cv2.imencode(".jpg", img_gray)  # type: ignore[attr-defined]
                if not ok:
                    raise RuntimeError("JPEG encode failed")

                files = {"image": (f"{image_id}.jpg", buffer.tobytes(), "image/jpeg")}
                data = {
                    "image_id": image_id,
                    "timestamp": time.time(),
                    "source": self.OS_friendly_name,
                    "mono": 1,
                }
                resp = session.post(self.upload_url, files=files, data=data, timeout=5)
                resp.raise_for_status()

                with self._mem_lock:
                    if image_id in self.ImageMem:
                        del self.ImageMem[image_id]

                elapsed_ms = (time.perf_counter() - start_ts) * 1000
                result = UploadResult(imageid=image_id, success=True, upload_time_ms=elapsed_ms)

                try:
                    self.upload_result_q.put_nowait(result)
                except Exception:
                    pass
        except Exception as e:
            import traceback
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
            import traceback
            tb_str = traceback.format_exc()
            try:
                self._error_q.put_nowait((threading.current_thread().name, e, tb_str))
            except Exception:
                pass
            return

