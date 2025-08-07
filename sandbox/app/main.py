import os
import uuid
import json
import time
import httpx
import aiofiles
from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    Request,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse, Response
from PIL import Image
import logging

from app.models import CropBox, ImageResponse, Settings

app = FastAPI(title="Product Image Processor")

# Set up logging - NB - this would be improved as structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch any unhandled exceptions."""
    logger.error(f"Unhandled exception occurred: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing your request",
            "detail": str(exc) if app.debug else "Please try again later",
        },
    )


def crop_image(image: Image.Image, crop_box: CropBox) -> Image.Image:
    """Crop an image using the provided crop box coordinates."""
    return image.crop(
        (
            crop_box.x,
            crop_box.y,
            crop_box.x + crop_box.width,
            crop_box.y + crop_box.height,
        )
    )


settings = Settings()

os.makedirs(settings.processed_images_dir, exist_ok=True)


async def process_image_in_background(
    image_id: str, image_data: bytes, product_data: dict
):
    """A background task to process the smart crop."""

    logger.info(f"Starting background processing for image {image_id}")
    try:
        # use connection pool if we want higher throughput
        async with httpx.AsyncClient() as client:
            ai_response = await client.post(
                settings.crop_endpoint_url,
                files={"image_file": ("image.jpg", image_data, "image/jpeg")},
                timeout=10.0,
            )
            ai_response.raise_for_status()
            crop_box_data = ai_response.json()["bounding_box"]
            crop_box = CropBox(**crop_box_data)

        image = Image.open(BytesIO(image_data))

        # Validate crop box against image dimensions
        crop_box.validate_against_image(image.width, image.height)

        cropped_image = image.crop(crop_box.to_pil_coords())

        output_path = f"{settings.processed_images_dir}/{image_id}.jpg"
        cropped_image.save(output_path, "JPEG")
        logger.info(
            f"Successfully processed and saved image {image_id} to {output_path}"
        )
    except Exception as e:
        logger.error(f"Error processing image {image_id}: {str(e)}")


@app.post("/images/manual-crop")
async def manual_crop(
    source_image: UploadFile = File(...),
    product_info: str = Form(...),
    crop_box: str = Form(...),
) -> ImageResponse:
    try:
        product_data = json.loads(product_info)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in product_info")

    try:
        crop_data = json.loads(crop_box)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in crop_box")

    # Use CropBox model for validation
    crop_box_model = CropBox(**crop_data)

    image_data = await source_image.read()
    image = Image.open(BytesIO(image_data))
    cropped_image = crop_image(image, crop_box_model)

    image_id = str(uuid.uuid4())
    output_path = f"{settings.processed_images_dir}/{image_id}.jpg"
    cropped_image.save(output_path, "JPEG")

    return ImageResponse(image_id=image_id, retrieval_url=f"/images/{image_id}.jpg")


@app.post("/images/smart-crop")
async def smart_crop(
    request: Request,
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(...),
    product_info: str = Form(...),
):
    try:
        product_data = json.loads(product_info)
        image_data = await source_image.read()
        image_id = str(uuid.uuid4())

        background_tasks.add_task(
            process_image_in_background, image_id, image_data, product_data
        )

        return JSONResponse(
            status_code=202,
            content={
                "image_id": image_id,
                "retrieval_url": f"/images/{image_id}.jpg",
                "status": "processing",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting image processing: {str(e)}"
        )


@app.get("/images/{image_path:path}")
async def get_image(image_path: str):
    full_image_path = f"{settings.processed_images_dir}/{image_path}"

    try:
        async with aiofiles.open(full_image_path, "rb") as f:
            content = await f.read()
            return Response(content=content, media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")


# Mock AI endpoint for smart-crop to call
@app.post("/mock-ai/find-main-object")
async def mock_ai_endpoint(image_file: UploadFile = File(...)):
    # Simulate a slow, non-blocking process
    time.sleep(2)
    _ = await image_file.read()
    return {"bounding_box": {"x": 50, "y": 50, "width": 150, "height": 150}}
