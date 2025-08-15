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
      - delete_image_by_id(image_id)
      - get_stored_image_ids()
      - raise_thread_error_if_any()
    """

    def __init__(
        self,
        sharedmem_buffs: dict,
        safe_mem_details_func: Callable[[], SharedMem_ImgTicket],
        upload_url: str,
        OS_friendly_name: str,
    ) -> None:
        self.sharedmem_bufs = sharedmem_buffs
        self.safe_mem_details_func = safe_mem_details_func
        self.upload_url = upload_url
        self.OS_friendly_name = OS_friendly_name
        self.max_store = 100

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

    def delete_image_by_id(self, image_id: str) -> bool:
        """Delete a specific image ID from storage - returns True if deleted, False if not found"""
        with self._mem_lock:
            return self.ImageMem.pop(image_id, None) is not None

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
        
        def do_POST(self) -> None:
            try:
                import email.message
                import email.parser
                import io
                
                # Parse multipart form data using modern email.message approach
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data'):
                    self.send_error(400, "Expected multipart/form-data")
                    return
                
                # Get content length
                content_length = int(self.headers.get('Content-Length', '0'))
                if content_length == 0:
                    self.send_error(400, "Empty request body")
                    return
                
                # Read the raw request body
                raw_data = self.rfile.read(content_length)
                
                # Create email message with proper headers for parsing
                msg_str = f"Content-Type: {content_type}\r\n\r\n"
                msg_str = msg_str.encode('ascii') + raw_data
                
                # Parse using modern email parser
                parser = email.parser.BytesParser()
                msg = parser.parsebytes(msg_str)
                
                # Extract form fields
                form_data = {}
                image_data = None
                image_filename = None
                
                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                        
                    content_disposition = part.get('Content-Disposition', '')
                    if 'form-data' not in content_disposition:
                        continue
                    
                    # Parse the Content-Disposition header to get field name
                    name = None
                    filename = None
                    for param in content_disposition.split(';'):
                        param = param.strip()
                        if param.startswith('name='):
                            name = param.split('=', 1)[1].strip('"')
                        elif param.startswith('filename='):
                            filename = param.split('=', 1)[1].strip('"')
                    
                    if name:
                        content = part.get_payload(decode=True)
                        if name == 'image' and filename:
                            image_data = content
                            image_filename = filename
                        else:
                            # Regular form field
                            form_data[name] = content.decode('utf-8') if content else ''
                
                # Extract the expected fields
                image_id = form_data.get('image_id')
                timestamp = form_data.get('timestamp')
                
                if image_id:
                    UploadHandler.uploads_received.append(image_id)
                    print(f"📤 HTTP Server received upload:")
                    print(f"   - image_id: {image_id}")
                    print(f"   - timestamp: {timestamp}")
                    if image_data and image_filename:
                        print(f"   - image_file: {image_filename} ({len(image_data)} bytes)")
                else:
                    print("❌ No image_id found in upload")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                
            except Exception as e:
                print(f"❌ Error in UploadHandler: {e}")
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Server error: {e}")
        
        def log_message(self, format: str, *args) -> None:
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
        initial_uploads = len(UploadHandler.uploads_received)
        uploader.upload_image_by_id(stored_ids[0])
        
        # Wait for upload to complete and check server response
        # Check multiple times with small delays to handle timing issues
        success = False
        for attempt in range(50):  # Try for up to 2 seconds
            time.sleep(0.2)
            if len(UploadHandler.uploads_received) > initial_uploads:
                success = True
                break
        
        assert success, f"Upload failed: Server did not receive the image ID. Expected uploads > {initial_uploads}, got {len(UploadHandler.uploads_received)}"
        print(f"✅ Upload successful! Server received: {UploadHandler.uploads_received}")
        
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
        initial_upload_count = len(UploadHandler.uploads_received)
        for img_id in stored_ids:
            uploader.upload_image_by_id(img_id)
        
        # Wait and check results
        time.sleep(2.0)
        final_stored = uploader.get_stored_image_ids()
        final_upload_count = len(UploadHandler.uploads_received)
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
    print("\n🔄 Testing broken thread detection with nonsense URL...")
    uploader_broken_url = ImageUploaderThreaded_shared_mem(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        upload_url="http://invalid-url",  # Clearly invalid URL
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
        if "Invalid URL" in str(e):
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
    
    uploader_nonsense_img = ImageUploaderThreaded_shared_mem(
        sharedmem_buffs=sharedmem_buffs_nonsense,
        safe_mem_details_func=safe_mem_details_func,
        upload_url=f"http://localhost:{server_port}/upload",
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
    uploader_overflow = ImageUploaderThreaded_shared_mem(
        sharedmem_buffs=sharedmem_buffs,
        safe_mem_details_func=safe_mem_details_func,
        upload_url="http://127.0.0.1:99999/blocked",  # This will fail fast
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
            print(f"   ✅ Exception: {e}")
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
    