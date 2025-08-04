from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class ProductInfo(BaseModel):
    product_id: str

class CropBox(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

class ImageResponse(BaseModel):
    image_id: str
    retrieval_url: str

class Settings(BaseSettings):
    processed_images_dir: str = Field(description="Directory to store processed images")
