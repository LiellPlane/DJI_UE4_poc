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
from pydantic import BaseModel, Field

# UploadResult removed - we no longer track upload results
# Network failures are expected and ignored, encoding failures crash immediately


class UploadRequest(BaseModel):
    """Pydantic model for validating upload request data"""
    image_id: str = Field(..., description="Unique identifier for the image")
    timestamp: float = Field(..., description="Unix timestamp when upload was initiated")
    source: str = Field(..., description="Source device/OS friendly name")
    

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
        
        # Give threads time to start up before constructor returns
        time.sleep(0.1)

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
                    
                    # Validate upload data using Pydantic model
                    upload_request = UploadRequest(
                        image_id=image_id,
                        timestamp=time.time(),
                        source=self.OS_friendly_name,
                    )
                    
                    resp = session.post(
                        self.upload_url, 
                        files=files, 
                        data=upload_request.model_dump(), 
                        timeout=2
                    )
                    resp.raise_for_status()
                    # Upload successful - no result tracking needed
                    
                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and 400 <= e.response.status_code < 500:
                        # 4xx = malformed request, CRASH
                        raise RuntimeError(f"Malformed upload request for {image_id}: HTTP {e.response.status_code}")
                    # 5xx = server error, ignore (network/server issue)
                    
                except requests.exceptions.InvalidURL:
                    # Invalid URL - report as a critical error
                    raise RuntimeError(f"Invalid URL for upload: {self.upload_url}")

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


if __name__ == "__main__":
    """Comprehensive test with real image data and HTTP server"""
    import time
    import numpy as np
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    from my_collections import SharedMem_ImgTicket
    
    print("🧪 Testing ImageUploaderThreaded_shared_mem with real data...")
    
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
    
    # Simple HTTP server to receive uploads
    class UploadHandler(BaseHTTPRequestHandler):
        uploads_received = []
        
        def do_POST(self):
            try:
                import cgi
                import io
                
                # Parse multipart form data properly
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data'):
                    self.send_error(400, "Expected multipart/form-data")
                    return
                
                # Create a proper environment for cgi.FieldStorage
                environ = {
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                    'CONTENT_LENGTH': self.headers.get('Content-Length', '0')
                }
                
                # Parse the form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ=environ
                )
                
                # Extract fields that match what _worker_loop sends
                image_id = form.getvalue('image_id')
                timestamp = form.getvalue('timestamp')
                source = form.getvalue('source')
                
                # Get the uploaded image file - check if field exists first
                image_field = None
                if 'image' in form:
                    image_field = form['image']
                
                if image_id:
                    UploadHandler.uploads_received.append(image_id)
                    print(f"📤 HTTP Server received upload:")
                    print(f"   - image_id: {image_id}")
                    print(f"   - timestamp: {timestamp}")
                    print(f"   - source: {source}")
                    if image_field is not None and hasattr(image_field, 'filename'):
                        image_data = image_field.file.read()
                        print(f"   - image_file: {image_field.filename} ({len(image_data)} bytes)")
                        image_field.file.seek(0)  # Reset file pointer
                else:
                    print("❌ No image_id found in upload")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                
            except Exception as e:
                print(f"❌ Error in UploadHandler: {e}")
                self.send_error(500, f"Server error: {e}")
        
        def log_message(self, format, *args):
            pass  # Suppress HTTP server logs
    
    # Start HTTP server in background
    server_port = 8765
    httpd = HTTPServer(('localhost', server_port), UploadHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"🌐 HTTP server started on localhost:{server_port}")
    
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
        
        # Create uploader with local HTTP server
        uploader = ImageUploaderThreaded_shared_mem(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            upload_url=f"http://localhost:{server_port}/upload",
            OS_friendly_name="test_pi",
            max_store=5
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
        
        # Test upload of image
        if stored_ids:
            print(f"\n📤 Testing upload of image: {stored_ids[0]}")
            initial_uploads = len(UploadHandler.uploads_received)
            uploader.upload_image_by_id(stored_ids[0])
            
            # Wait for upload to complete and check server response
            # Check multiple times with small delays to handle timing issues
            success = False
            for attempt in range(10):  # Try for up to 2 seconds
                time.sleep(0.2)
                if len(UploadHandler.uploads_received) > initial_uploads:
                    success = True
                    break
            
            if success:
                print(f"✅ Upload successful! Server received: {UploadHandler.uploads_received}")
            else:
                print("❌ Upload failed: Server did not receive the image ID")
                print(f"   Debug: Expected uploads > {initial_uploads}, got {len(UploadHandler.uploads_received)}")
        else:
            print("❌ No images to upload")
        
        # Test multiple captures and uploads
        print("\n🔄 Testing multiple operations...")
        for i in range(3):
            uploader.trigger_capture()
            time.sleep(0.05)
        
        stored_ids = uploader.get_stored_image_ids()
        print(f"✅ Multiple captures: {len(stored_ids)} images stored")
        
        # Upload all stored images
        for img_id in stored_ids:
            uploader.upload_image_by_id(img_id)
        
        # Wait and check results
        time.sleep(2.0)
        final_stored = uploader.get_stored_image_ids()
        print(f"✅ After uploads: {len(final_stored)} images remaining")
        print(f"✅ Total uploads received by server: {len(UploadHandler.uploads_received)}")
        
        # Final health check
        uploader.raise_thread_error_if_any()
        print("✅ Threads still healthy after all operations")
        
        print(f"\n🎉 Comprehensive test completed!")
        print(f"   📊 Images uploaded: {len(UploadHandler.uploads_received)}")
        print(f"   📊 Server responses: {UploadHandler.uploads_received}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        httpd.shutdown()
        print("🧹 HTTP server stopped")

    # Test broken thread detection with nonsense URL
    try:
        print("\n🔄 Testing broken thread detection with nonsense URL...")
        uploader_broken_url = ImageUploaderThreaded_shared_mem(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            upload_url="http://invalid-url",  # Clearly invalid URL
            OS_friendly_name="test_pi",
            max_store=5
        )
        uploader_broken_url.trigger_capture()
        time.sleep(0.1)  # Let capture process
        stored_ids = uploader_broken_url.get_stored_image_ids()
        if stored_ids:
            uploader_broken_url.upload_image_by_id(stored_ids[0])
        time.sleep(1.0)
        # Expecting an exception here due to nonsense URL
        uploader_broken_url.raise_thread_error_if_any()
        print("⚠️  Nonsense URL test failed: No thread errors detected")
        print("   ℹ️  This error is hard to detect because network failures are")
        print("   ℹ️  intentionally ignored by design - only 4xx client errors crash")
    except RuntimeError as e:
        if "Invalid URL" in str(e):
            print(f"✅ Nonsense URL test passed: {e}")
        else:
            print(f"❌ Nonsense URL test failed: Unexpected error type: {e}")
    except Exception as e:
        print(f"❌ Nonsense URL test failed: Unexpected exception: {e}")

    # Test broken thread detection with nonsense image
    try:
        print("\n🔄 Testing broken thread detection with nonsense image...")
        def create_nonsense_image(width=640, height=480):
            """Create a nonsense image that will fail debuffering"""
            img = np.zeros((height, width), dtype=np.uint8)
            img[0, 0:16] = np.random.randint(0, 256, size=16, dtype=np.uint8)  # Random bytes
            return img
        
        nonsense_img = create_nonsense_image()
        nonsense_img_bytes = nonsense_img.tobytes()
        sharedmem_buffs_nonsense = {0: MockSharedMem(nonsense_img_bytes)}
        
        uploader_nonsense_img = ImageUploaderThreaded_shared_mem(
            sharedmem_buffs=sharedmem_buffs_nonsense,
            safe_mem_details_func=safe_mem_details_func,
            upload_url=f"http://localhost:{server_port}/upload",
            OS_friendly_name="test_pi",
            max_store=5
        )
        uploader_nonsense_img.trigger_capture()
        time.sleep(0.1)  # Let capture process
        # Expecting an exception here due to nonsense image
        uploader_nonsense_img.raise_thread_error_if_any()
        print("❌ Nonsense image test failed: No thread errors detected")
    except RuntimeError as e:
        if "decode" in str(e):
            print(f"✅ Nonsense image test passed: {e}")
        else:
            print(f"❌ Nonsense image test failed: Unexpected error type: {e}")
    except Exception as e:
        print(f"❌ Nonsense image test failed: Unexpected exception: {e}")

    # Test queue overflow protection
    try:
        print("\n🔄 Testing upload queue overflow protection...")
        
        # Create uploader for overflow testing
        uploader_overflow = ImageUploaderThreaded_shared_mem(
            sharedmem_buffs=sharedmem_buffs,
            safe_mem_details_func=safe_mem_details_func,
            upload_url="http://127.0.0.1:99999/blocked",  # This will fail fast
            OS_friendly_name="test_pi",
            max_store=10
        )
        
        # Capture one image to have something to upload
        uploader_overflow.trigger_capture()
        time.sleep(0.1)
        stored_ids = uploader_overflow.get_stored_image_ids()
        print(f"   ✅ Captured {len(stored_ids)} images for testing")
        
        if not stored_ids:
            print("   ❌ No images captured for overflow test")
            raise Exception("No images available for overflow test")
        
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
                print(f"   ✅ Exception: {e}")
                break
        
        if not queue_full_detected:
            print(f"   ⚠️  Queue overflow not detected after {upload_attempts} rapid uploads")
            print("   ℹ️  Worker thread may be processing too quickly")
        
        print("   ✅ Queue blast test completed")
        
    except Exception as e:
        if "Full" in str(e) or "queue" in str(e).lower():
            print(f"✅ Queue overflow test passed: {e}")
        else:
            print(f"❌ Queue overflow test failed: {e}")
    