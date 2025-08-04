# import pytest
# from pydantic import ValidationError
# from app.models import ProductInfo, CropBox, ImageResponse


# class TestProductInfo:
#     """Tests for ProductInfo model."""

#     def test_valid_product_info(self):
#         """Test creating valid ProductInfo."""
#         product = ProductInfo(product_id="test-123")
#         assert product.product_id == "test-123"

#     def test_missing_product_id(self):
#         """Test ProductInfo validation with missing product_id."""
#         with pytest.raises(ValidationError):
#             ProductInfo()


# class TestCropBox:
#     """Tests for CropBox model."""

#     def test_valid_crop_box(self):
#         """Test creating valid CropBox."""
#         crop_box = CropBox(x=10, y=20, width=100, height=150)
#         assert crop_box.x == 10
#         assert crop_box.y == 20
#         assert crop_box.width == 100
#         assert crop_box.height == 150

#     def test_negative_coordinates(self):
#         """Test CropBox validation with negative coordinates."""
#         with pytest.raises(ValidationError):
#             CropBox(x=-1, y=20, width=100, height=150)

#         with pytest.raises(ValidationError):
#             CropBox(x=10, y=-1, width=100, height=150)

#     def test_zero_dimensions(self):
#         """Test CropBox validation with zero dimensions."""
#         with pytest.raises(ValidationError):
#             CropBox(x=10, y=20, width=0, height=150)

#         with pytest.raises(ValidationError):
#             CropBox(x=10, y=20, width=100, height=0)

#     def test_missing_fields(self):
#         """Test CropBox validation with missing required fields."""
#         with pytest.raises(ValidationError):
#             CropBox(x=10, y=20)  # Missing width and height


# class TestImageResponse:
#     """Tests for ImageResponse model."""

#     def test_valid_image_response(self):
#         """Test creating valid ImageResponse."""
#         response = ImageResponse(
#             image_id="test-123",
#             retrieval_url="/images/test-123.jpg"
#         )
#         assert response.image_id == "test-123"
#         assert response.retrieval_url == "/images/test-123.jpg"

#     def test_missing_fields(self):
#         """Test ImageResponse validation with missing fields."""
#         with pytest.raises(ValidationError):
#             ImageResponse(image_id="test-123")  # Missing retrieval_url
