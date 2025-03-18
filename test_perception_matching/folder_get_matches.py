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
from pathlib import Path
from dataclasses import dataclass
from qdrant_client.http.models.models import ScoredPoint


QDRANT_COLLECTION_NAME = "everything"

# Detect operating system and set appropriate paths
if platform.system() == "Darwin":  # macOS
    IMAGES_TO_MATCH_PATH = Path("/Users/liell_p/match_images")
    OUTPUT_PATH = Path("/Users/liell_p/match_images_output")
elif platform.system() == "Windows":
    IMAGES_TO_MATCH_PATH = Path(r"D:\temp_match_imgs\fart")
    OUTPUT_PATH = Path("D:\match_images_output")
else:  # Linux or other OS
    IMAGES_TO_MATCH_PATH = Path("/tmp/temp_match_imgs/")
    OUTPUT_PATH = Path("/tmp/match_images_output")

@dataclass
class ScoredPointWithFlip:
    """
    Simple wrapper for ScoredPoint that adds information about whether
    the match was found with a flipped version of the image.
    """
    point: ScoredPoint
    is_flipped: bool = False

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
            # Flip the image horizontally (mirror effect)
            img_flipped = cv2.flip(img, 1)
            embedding = generate_embeddings.create_image_embedding(
                img, 
                params=GRABBED_EMBEDDING_PARAMS,
                mask=None
            )
            embedding_flipped = generate_embeddings.create_image_embedding(
                img_flipped, 
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
            search_result_flipped = client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=embedding_flipped,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            # Create a list of all results with flipped information
            all_results = [ScoredPointWithFlip(pt, is_flipped=True) for pt in search_result_flipped]
            all_results.extend([ScoredPointWithFlip(pt, is_flipped=False) for pt in search_result])
            
            # Sort by score and take top 10
            top_10 = sorted(all_results, key=lambda x: x.point.score, reverse=True)[:10]
            output[image_path] = top_10

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
            if res.is_flipped:
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
    