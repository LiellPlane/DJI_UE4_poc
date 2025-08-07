from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProductInfo(BaseModel):
    product_id: str


class CropBox(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    def validate_against_image(self, image_width: int, image_height: int):
        """Validate crop box dimensions against image size."""
        if (self.x + self.width) > image_width:
            raise ValueError("Crop box exceeds image width")
        if (self.y + self.height) > image_height:
            raise ValueError("Crop box exceeds image height")

    def to_pil_coords(self):
        """Convert to PIL crop coordinates (left, top, right, bottom)."""
        return (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )


class ImageResponse(BaseModel):
    image_id: str
    retrieval_url: str


class Settings(BaseSettings):
    processed_images_dir: str
    base_url: str
    crop_endpoint_url: str

    # DynamoDB settings
    dynamodb_endpoint_url: str
    dynamodb_table_name: str
    rate_limit_requests_per_hour: int
