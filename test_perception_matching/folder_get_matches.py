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


USE_DINO = True # are we using dino model for embeddings or our custom one
if USE_DINO is True:
    import get_image_embedding_DINO
QDRANT_COLLECTION_NAME = "food"

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

def create_white_square(size=300):
    """Create a white square image of specified size."""
    return np.ones((size, size, 3), dtype=np.uint8) * 255

def create_red_square(size=300):
    """Create a white square image of specified size."""
    return np.ones((size, size, 3), dtype=np.uint8) * [0,0,255]

def main():
    image_paths = get_batch_embeddings.get_image_filepaths_from_folders(
        [IMAGES_TO_MATCH_PATH]
    )
    output={}
    for index, image_path in enumerate(image_paths):
        try:
            print(f"processing {index} of {len(image_paths)}")
            if USE_DINO:
                embedding, embedding_flipped = get_image_embedding_DINO.get_image_embedding(image_path)
            else:
                embedding, embedding_flipped = get_image_embedding.get_image_embedding(client, image_path, QDRANT_COLLECTION_NAME)

            sorted_files = get_image_embedding.get_closest_match(client, embedding, embedding_flipped, QDRANT_COLLECTION_NAME)

            output[image_path] = sorted_files
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    indexer=0
    for index, (image_path, search_result) in enumerate(output.items()):
        # Copy the original image

        
        # Insert white squares as visual buffer
        white_square = create_white_square(300)
        for i in range(5):  # Save 5 white squares
            white_square_path = OUTPUT_PATH / f"{indexer:004d}.jpg"
            cv2.imwrite(str(white_square_path), white_square)
            indexer+=1
        shutil.copy2(image_path, OUTPUT_PATH / f"{indexer:004d}.jpg")
        indexer+=1
        if search_result[0].point.score > 0.95:
            red_square = create_red_square(300)
            cv2.imwrite(str(OUTPUT_PATH / f"{indexer:004d}.jpg"), red_square)
            indexer+=1
        # Copy search results
        for res in search_result:
            if not res.is_flipped:
                # load the image and flip it
                img = cv2.imread(res.point.payload["filename"])
                img = cv2.flip(img, 1)
                cv2.imwrite(str(OUTPUT_PATH / f"{indexer:004d}.jpg"), img)
                indexer+=1
            else:
                shutil.copy2(res.point.payload["filename"], OUTPUT_PATH / f"{indexer:004d}.jpg")
                indexer+=1


if __name__ == '__main__':
    main()
    