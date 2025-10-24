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
from test_async_qdrant import AsyncClosestMatchHandler, ClosestMatchResult
from qdrant_client import AsyncQdrantClient
USE_DINO = True # are we using dino model for embeddings or our custom one
if USE_DINO is True:
    import get_image_embedding_DINO

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

def extract_region_from_square(image: np.ndarray, square: Square) -> np.ndarray:
    """
    Extract a region from an image based on a Square object.
    
    Parameters
    ----------
    image : np.ndarray
        The input image
    square : Square
        The Square object defining the region to extract
        
    Returns
    -------
    np.ndarray
        The extracted region
    """
    # Ensure coordinates are within image bounds
    x = max(0, min(square.x, image.shape[1] - 1))
    y = max(0, min(square.y, image.shape[0] - 1))
    side = min(square.side, image.shape[1] - x, image.shape[0] - y)
    
    # Extract the region
    region = image[y:y+side, x:x+side]
    return region

async def main():
    # Create output directory if it doesn't exist
    QDRANT_COLLECTION_NAME = "everything"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    async_client =  AsyncQdrantClient("localhost")
    sync_client = get_sequence_images_qdrant.get_qdrant_client()
    async_handler = AsyncClosestMatchHandler(
        collection_name=QDRANT_COLLECTION_NAME,
        real_client=async_client
        )
    

    vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
        sync_client,
        collection_name=QDRANT_COLLECTION_NAME,
        limit=1
    )
    if USE_DINO is True:
        GRABBED_EMBEDDING_PARAMS = None
    else:
        GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))
    # Get all images to process
    image_paths = get_batch_embeddings.get_image_filepaths_from_folders([IMAGES_TO_MATCH_PATH])
    
    # Process each image
    for index, image_path in enumerate(image_paths):
        try:
            print(f"Processing {index + 1} of {len(image_paths)}: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
                
            # Create canvas and sample squares
            canvas = Canvas(image.shape[1], image.shape[0])
            squares = sample_square_regions(
                canvas=canvas,
                max_side_ratio=0.8,
                min_side_ratio=0.1,
                n_sizes=5,
                min_distance_ratio=0.2,
                )
            
            # Process each square region
            embeddings:dict [Square, np.ndarray] = {}
            for sqr_index, square in enumerate(squares):
                # Extract the region
                region = extract_region_from_square(image, square)
                # if sqr_index % 100 == 0:
                #     cv2.imshow("Region", region)
                #     cv2.waitKey(1)
                if USE_DINO:
                    embedding, embedding_flipped = get_image_embedding_DINO.get_image_embedding(image_path,img_in_memory=region )
                else:    
                    # Get embeddings for the region
                    embedding, embedding_flipped = get_image_embedding.get_image_embedding(
                        sync_client, 
                        image_path, 
                        QDRANT_COLLECTION_NAME, 
                        img_in_memory=region, 
                        GRABBED_EMBEDDING_PARAMS=GRABBED_EMBEDDING_PARAMS
                    )
                embeddings[square] = embedding
                # Process the embeddings as needed
            res = await async_handler.process_embeddings(embeddings, 1)
            
            # Get the best match for each square
            best_square: Square = list((res.keys()))[0] # start anywhere
            best_score: float = -1
            filtered_res = {}
            for square, embedding in res.items():
                for score, filepath in zip(embedding.scores, embedding.filepaths):
                    if score > best_score:
                        best_score = score
                        best_square = square
                        best_image = filepath
                    

            # TODO this doesnt quite work so just for testing - expects only one match per square
            sorted_results = dict(sorted(res.items(), key=lambda x: max(x[1].scores), reverse=False))
            multiple_match_image = image.copy()
            for square, embedding in sorted_results.items():
                if len(embedding.filepaths) > 1:
                    raise Exception(f"Multiple matches for square {square} - logic not expecting this")
                for score, filepath in zip(embedding.scores, embedding.filepaths):
                    matched_image = cv2.imread(filepath)
                    matched_image = cv2.resize(matched_image, (square.side, square.side))
                    multiple_match_image[square.y:square.y+square.side, square.x:square.x+square.side] = matched_image
            
            # cv2.imshow("Multiple Match Image", cv2.resize(multiple_match_image, (800, 800)))
            # key = cv2.waitKey(0)
            
            # now load the best match
            print(f"Best image: {best_image}, best score: {best_score}")
            best_image = cv2.imread(best_image)

            # Resize the best match image to match the square's dimensions
            best_image = cv2.resize(best_image, (best_square.side, best_square.side))
            
            image_with_best_match = image.copy()
            image_with_best_match[best_square.y:best_square.y+best_square.side, best_square.x:best_square.x+best_square.side] = best_image
            cv2.imwrite(f"{OUTPUT_PATH}/best_match_{index}.jpg", multiple_match_image)
            while True:
                cv2.imshow("Image with Best Match", cv2.resize(multiple_match_image, (800, 800)))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    break
                cv2.imshow("Image with Best Match", cv2.resize(image, (800, 800)))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    break
            cv2.destroyAllWindows()

                

        # TODO: Process the embeddings as needed
        # print(f"Processed square at ({square.x}, {square.y}) with size {square.side}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    asyncio.run(main())
    