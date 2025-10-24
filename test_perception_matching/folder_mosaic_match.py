import cv2
import numpy as np
import os
import get_sequence_images_qdrant
import generate_embeddings
import get_batch_embeddings
import json
import shutil
import cv2
import platform
import get_image_embedding
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import tempfile
USE_DINO = True # are we using dino model for embeddings or our custom one
if USE_DINO is True:
    import get_image_embedding_DINO
QDRANT_COLLECTION_NAME = "everything"

# Detect operating system and set appropriate paths
if platform.system() == "Darwin":  # macOS
    IMAGES_TO_MATCH_PATH = Path("/Users/liell_p/match_images")
    OUTPUT_PATH = Path("/Users/liell_p/match_images_output")
elif platform.system() == "Windows":
    IMAGES_TO_MATCH_PATH = Path("D:\match_images")
    OUTPUT_PATH = Path("D:\match_images_output")
else:  # Linux or other OS
    IMAGES_TO_MATCH_PATH = Path("/tmp/temp_match_imgs/")
    OUTPUT_PATH = Path("/tmp/match_images_output")

client = get_sequence_images_qdrant.get_qdrant_client()
vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
    client,
    collection_name=QDRANT_COLLECTION_NAME,
    limit=1
)
# ensure using same embeddings by grabbing it straight from qdrant collection
# GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))

def get_image_segments(image: np.ndarray, grid_size: Tuple[int, int] = (30, 30)) -> List[np.ndarray]:
    """Divide an image into segments based on grid size."""
    height, width = image.shape[:2]
    segment_height = height // grid_size[0]
    segment_width = width // grid_size[1]
    
    segments = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y_start = i * segment_height
            y_end = (i + 1) * segment_height
            x_start = j * segment_width
            x_end = (j + 1) * segment_width
            segment = image[y_start:y_end, x_start:x_end]
            segments.append(segment)
    return segments

def create_mosaic(image_path: str, grid_size: Tuple[int, int] = (20, 20), output_size: Tuple[int, int] = (3000, 3000)) -> np.ndarray:
    """Create a mosaic from an image using the Qdrant database for matching."""
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get segments
    segments = get_image_segments(img, grid_size)
    
    # Create output mosaic
    mosaic_height = output_size[0]
    mosaic_width = output_size[1]
    segment_height = mosaic_height // grid_size[0]
    segment_width = mosaic_width // grid_size[1]
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    
    # Create temporary directory for segment files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each segment
        for i, segment in enumerate(segments):
            try:
                # Save segment to temporary file
                temp_segment_path = os.path.join(temp_dir, f"segment_{i}.jpg")
                cv2.imwrite(temp_segment_path, segment)
                
                # Get embedding for the segment
                if USE_DINO:
                    embedding, embedding_flipped = get_image_embedding_DINO.get_image_embedding(temp_segment_path)
                else:
                    embedding, embedding_flipped = get_image_embedding.get_image_embedding(client, temp_segment_path, QDRANT_COLLECTION_NAME)
                
                # Get closest match
                matches = get_image_embedding.get_closest_match(client, embedding, embedding_flipped, QDRANT_COLLECTION_NAME, limit=1)
                if not matches:
                    continue
                    
                # Load the matched image
                matched_image_path = matches[0].point.payload["filename"]
                matched_img = cv2.imread(matched_image_path)
                if matched_img is None:
                    continue
                    
                # Resize matched image to fill the segment space completely
                matched_img = cv2.resize(matched_img, (segment_width, segment_height))
                
                # Calculate position in mosaic
                row = i // grid_size[1]
                col = i % grid_size[1]
                y_start = row * segment_height
                y_end = (row + 1) * segment_height
                x_start = col * segment_width
                x_end = (col + 1) * segment_width
                
                # Place in mosaic
                mosaic[y_start:y_end, x_start:x_end] = matched_img
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                continue
    
    return mosaic

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Get all images to process
    image_paths = get_batch_embeddings.get_image_filepaths_from_folders([IMAGES_TO_MATCH_PATH])
    
    # Process each image
    for index, image_path in enumerate(image_paths):
        try:
            print(f"Processing {index + 1} of {len(image_paths)}: {image_path}")
            
            # Create mosaic
            mosaic = create_mosaic(image_path)
            
            # Save mosaic
            output_path = OUTPUT_PATH / f"mosaic_{index:04d}.jpg"
            cv2.imwrite(str(output_path), mosaic)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    main()
    