FROM python:3.11

# Install UV
RUN pip install uv

WORKDIR /app

# Set environment variables
ENV PROCESSED_IMAGES_DIR=/app/processed_images

# Copy the application
COPY . /app

# Install dependencies using UV
RUN uv sync --frozen

EXPOSE 8000

CMD uv run uvicorn app.main:app --host 0.0.0.0 --port 8000