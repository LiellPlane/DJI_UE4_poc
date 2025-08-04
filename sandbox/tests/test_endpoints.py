import json
import os
from unittest.mock import patch, Mock
from app.models import Settings, CropBox
from app.main import crop_image
from PIL import Image
from io import BytesIO


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
                "crop_box": json.dumps(crop_box)
            }
        )

        assert response.status_code == 422  # Missing required file


# class TestSmartCrop:
#     """Tests for the smart crop endpoint."""

#     @patch('app.main.requests.post')
#     def test_smart_crop_success(self, mock_post, client, sample_image):
#         """Test successful smart crop with mocked AI response."""
#         # Mock the AI response
#         mock_response = Mock()
#         mock_response.json.return_value = {
#             "bounding_box": {"x": 50, "y": 50, "width": 100, "height": 100}
#         }
#         mock_response.raise_for_status.return_value = None
#         mock_post.return_value = mock_response

#         product_info = {"product_id": "test-123"}

#         response = client.post(
#             "/images/smart-crop",
#             files={"source_image": ("test.jpg", sample_image, "image/jpeg")},
#             data={"product_info": json.dumps(product_info)}
#         )

#         assert response.status_code == 202
#         data = response.json()
#         assert "image_id" in data
#         assert "retrieval_url" in data
#         assert data["status"] == "processing"

#     def test_smart_crop_invalid_product_info(self, client, sample_image):
#         """Test smart crop with invalid product info."""
#         response = client.post(
#             "/images/smart-crop",
#             files={"source_image": ("test.jpg", sample_image, "image/jpeg")},
#             data={"product_info": "invalid-json"}
#         )

#         assert response.status_code == 500  # JSON decode error


# class TestImageRetrieval:
#     """Tests for image retrieval endpoint."""

#     def test_get_nonexistent_image(self, client):
#         """Test retrieving a non-existent image."""
#         response = client.get("/images/nonexistent.jpg")
#         assert response.status_code == 500  # File not found

#     # Note: Testing successful image retrieval would require
#     # setting up actual processed images in the test environment


# class TestMockAIEndpoint:
#     """Tests for the mock AI endpoint."""

#     def test_mock_ai_endpoint(self, client, sample_image):
#         """Test the mock AI endpoint."""
#         response = client.post(
#             "/mock-ai/find-main-object",
#             files={"image_file": ("test.jpg", sample_image, "image/jpeg")}
#         )

#         assert response.status_code == 200
#         data = response.json()
#         assert "bounding_box" in data
#         bbox = data["bounding_box"]
#         assert all(key in bbox for key in ["x", "y", "width", "height"])
#         assert bbox == {"x": 50, "y": 50, "width": 150, "height": 150}

#     def test_mock_ai_endpoint_no_file(self, client):
#         """Test mock AI endpoint without file."""
#         response = client.post("/mock-ai/find-main-object")
#         assert response.status_code == 422  # Missing required file
