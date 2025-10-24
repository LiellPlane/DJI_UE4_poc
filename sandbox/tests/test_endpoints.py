import json
import os
import time
from app.models import Settings, CropBox
from app.main import crop_image
from PIL import Image
from io import BytesIO


def wait_for_file(file_path, timeout=10):
    """Wait for file to exist with timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(0.5)
    return False


class TestManualCrop:
    """Tests for the manual crop endpoint."""

    def test_manual_crop_success(self, client, sample_image, cleanup_processed_images):
        """Test successful manual crop."""
        product_info = {"product_id": "test-123"}
        crop_box = {"x": 10, "y": 10, "width": 50, "height": 50}

        response = client.post(
            "/images/manual-crop",
            files={"source_image": ("test.jpg", sample_image, "image/jpeg")},
            data={
                "product_info": json.dumps(product_info),
                "crop_box": json.dumps(crop_box),
            },
        )
        assert response.status_code == 200

        # now check that the image is saved in the processed images directory
        settings = Settings()
        assert os.path.exists(
            f"{settings.processed_images_dir}/{response.json()['image_id']}.jpg"
        )

        # now check the image retrieval url works - this should really be its
        # own test, but for now we'll just check the image is there
        image_response = client.get(f"/images/{response.json()['image_id']}.jpg")
        assert image_response.status_code == 200
        assert len(image_response.content) > 0
        assert image_response.headers["content-type"] == "image/jpeg"

        sample_image.seek(0)
        original_image = Image.open(sample_image)
        expected_cropped = crop_image(original_image, CropBox(**crop_box))

        # Convert expected image to bytes for comparison
        expected_bytes = BytesIO()
        expected_cropped.save(expected_bytes, format="JPEG")
        expected_bytes = expected_bytes.getvalue()

        assert image_response.content == expected_bytes

    def test_manual_crop_invalid_json(self, client, sample_image):
        """Test manual crop with invalid JSON in form data."""
        crop_box = {"x": 10, "y": 10, "width": 50, "height": 50}

        response = client.post(
            "/images/manual-crop",
            files={"source_image": ("test.jpg", sample_image, "image/jpeg")},
            data={"product_info": "invalid-json", "crop_box": json.dumps(crop_box)},
        )
        assert response.status_code == 422
        assert "Invalid JSON in product_info" in response.json()["detail"]

    def test_manual_crop_missing_file(self, client):
        """Test manual crop without uploading a file."""
        product_info = {"product_id": "test-123"}
        crop_box = {"x": 10, "y": 10, "width": 50, "height": 50}

        response = client.post(
            "/images/manual-crop",
            data={
                "product_info": json.dumps(product_info),
                "crop_box": json.dumps(crop_box),
            },
        )

        assert response.status_code == 422  # Missing required file


class TestSmartCrop:
    """
    Tests for the smart crop endpoint

    this is more an integration test but we will leave it here due to time constraints
    """

    def test_smart_crop_with_mock_ai(self, client, mock_ai_server, sample_image):
        """Test smart crop with mock AI server."""

        product_info = {"product_id": "test-integration"}

        response = client.post(
            "/images/smart-crop",
            files={"source_image": ("test.jpg", sample_image.getvalue(), "image/jpeg")},
            data={"product_info": json.dumps(product_info)},
        )

        assert response.status_code == 202
        data = response.json()
        assert "image_id" in data
        assert "retrieval_url" in data
        assert data["status"] == "processing"

        settings = Settings()
        processed_image_path = f"{settings.processed_images_dir}/{data['image_id']}.jpg"
        assert wait_for_file(
            processed_image_path, timeout=10
        ), "Background processing timed out"
