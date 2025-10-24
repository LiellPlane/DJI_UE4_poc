FROM python:3.11

# Install UV
RUN pip install uv

WORKDIR /app

# Set environment variables
ENV PROCESSED_IMAGES_DIR=/app/processed_images
ENV BASE_URL=http://127.0.0.1:8000
ENV CROP_ENDPOINT_URL=http://127.0.0.1:8000/mock-ai/find-main-object
ENV DYNAMODB_ENDPOINT_URL=http://localstack:4566
ENV DYNAMODB_TABLE_NAME=product_images
ENV RATE_LIMIT_REQUESTS_PER_HOUR=20

# Copy the application
COPY . /app

# Install dependencies using UV
RUN uv sync --frozen

EXPOSE 8000

CMD uv run uvicorn app.main:app --host 0.0.0.0 --port 8000