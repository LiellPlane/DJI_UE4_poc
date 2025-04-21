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
import asyncio
from square_sampling import sample_square_regions, Canvas, Square
QDRANT_COLLECTION_NAME = "fishwars"

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


async def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sync_client = get_sequence_images_qdrant.get_qdrant_client()


    vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
    sync_client,
    collection_name=QDRANT_COLLECTION_NAME,
    limit=1
    )
    GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))
    # Get all images to process
    image_paths = get_batch_embeddings.get_image_filepaths_from_folders([IMAGES_TO_MATCH_PATH])
    
    # Process each image
    for index, image_path in enumerate(image_paths):
        try:
            print(f"Processing {index + 1} of {len(image_paths)}: {image_path}")
            image = cv2.imread(image_path)
            cv2.imshow("Image", image)
            squares = sample_square_regions(canvas=Canvas(image.shape[1], image.shape[0]))
            for square in squares:
                print(square)
                print("this can be done in parallel")
                embedding, embedding_flipped = get_image_embedding.get_image_embedding(sync_client, image_path, QDRANT_COLLECTION_NAME, img_in_memory=image, GRABBED_EMBEDDING_PARAMS= GRABBED_EMBEDDING_PARAMS)

            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    asyncio.run(main())
    