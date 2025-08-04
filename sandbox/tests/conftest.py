import pytest
import tempfile
import os
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

# Set env variable - this can be done a bit tidier with a fixture

os.environ["PROCESSED_IMAGES_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed_images"
)

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as client:
        yield client


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
        original_dir = "/app/processed_images"
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
