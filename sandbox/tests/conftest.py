import pytest
import tempfile
import os
import threading
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

# Set env variables - this can be done a bit tidier with a fixture

os.environ["PROCESSED_IMAGES_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed_images"
)
os.environ["CROP_ENDPOINT_URL"] = "http://127.0.0.1:8001/mock-ai/find-main-object"
os.environ["BASE_URL"] = "http://127.0.0.1:8000"

# DynamoDB settings for testing
os.environ["DYNAMODB_ENDPOINT_URL"] = "http://localhost:4566"  # LocalStack default
os.environ["DYNAMODB_TABLE_NAME"] = "rate_limits"
os.environ["RATE_LIMIT_REQUESTS_PER_HOUR"] = "20"

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def mock_ai_server():
    """Spin up a server to mock the external mock-ai endpoint."""

    class MockAIHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/mock-ai/find-main-object":
                time.sleep(2)
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length > 0:
                    _ = self.rfile.read(content_length)
                response = {
                    "bounding_box": {"x": 50, "y": 50, "width": 150, "height": 150}
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    port = 8001
    server = HTTPServer(("127.0.0.1", port), MockAIHandler)

    def run_server():
        server.serve_forever()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(1)  # Give server time to start
    yield f"http://127.0.0.1:{port}"

    server.shutdown()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple test image
    image = Image.new("RGB", (200, 200), color="red")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    return image_bytes


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory for processed images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the processed images directory
        os.environ["PROCESSED_IMAGES_DIR"] = temp_dir
        yield temp_dir
        # Clean up
        if "PROCESSED_IMAGES_DIR" in os.environ:
            del os.environ["PROCESSED_IMAGES_DIR"]


@pytest.fixture
def cleanup_processed_images():
    """Clean up the processed images directory after tests."""
    import shutil

    yield  # Run the test first

    # Clean up after test
    processed_dir = os.environ.get("PROCESSED_IMAGES_DIR")
    if processed_dir and os.path.exists(processed_dir) and os.path.isdir(processed_dir):
        # Remove all files but keep the directory
        for filename in os.listdir(processed_dir):
            file_path = os.path.join(processed_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
