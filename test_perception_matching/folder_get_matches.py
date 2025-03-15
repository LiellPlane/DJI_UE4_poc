import cv2
import numpy as np
import os
import get_sequence_images_qdrant
import generate_embeddings
import get_batch_embeddings
import json
import shutil
import cv2
from pathlib import Path
QDRANT_COLLECTION_NAME = "everything"
IMAGES_TO_MATCH_PATH = Path(r"D:\temp_match_imgs\naughty")
OUTPUT_PATH = Path("D:\match_images_output")

client = get_sequence_images_qdrant.get_qdrant_client()
vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
    client,
    collection_name=QDRANT_COLLECTION_NAME,
    limit=1
    )
# ensure using same embeddings by grabbing it straight from qdrant collection
GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))

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
            img = cv2.imread(image_path)
            embedding = generate_embeddings.create_image_embedding(
                img, 
                params=GRABBED_EMBEDDING_PARAMS,
                mask=None
            )
            search_result = client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=embedding,
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            output[image_path] = search_result
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
        if search_result[0].score > 0.95:
            red_square = create_red_square(300)
            cv2.imwrite(str(OUTPUT_PATH / f"{indexer:004d}.jpg"), red_square)
            indexer+=1
        # Copy search results
        for res in search_result:

            shutil.copy2(res.payload["filename"], OUTPUT_PATH / f"{indexer:004d}.jpg")
            indexer+=1


if __name__ == '__main__':
    main()
    