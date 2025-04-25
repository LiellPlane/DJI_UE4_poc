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
from qdrant_utils import delete_point, clone_collection, wait_for_collection_ready
from test_async_qdrant import AsyncClosestMatchHandler
import asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient

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


# client = get_sequence_images_qdrant.get_qdrant_client()
# vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
#     client,
#     collection_name=QDRANT_COLLECTION_NAME,
#     limit=1
#     )
# # ensure using same embeddings by grabbing it straight from qdrant collection
# GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))

def create_white_square(size=300):
    """Create a white square image of specified size."""
    return np.ones((size, size, 3), dtype=np.uint8) * 255

def create_red_square(size=300):
    """Create a white square image of specified size."""
    return np.ones((size, size, 3), dtype=np.uint8) * [0,0,255]


@dataclass
class ImageEmbedding:
    embedding: np.ndarray
    embedding_flipped: np.ndarray
    lerp_steps: list[np.ndarray]
    filename: str

@dataclass
class ImageWithScore:
    image_path: str
    score: float

def lerp(start: np.ndarray, end: np.ndarray, steps: int, debug: bool = False, use_slerp: bool = True) -> np.ndarray:
    """
    Interpolate between two high-dimensional embeddings using a combination of techniques.
    
    Args:
        start: Starting embedding vector
        end: Ending embedding vector
        steps: Number of interpolation steps
        debug: If True, print debug information
        use_slerp: If True, use spherical linear interpolation, else use simple linear interpolation
        
    Returns:
        Array of interpolated vectors
    """
    # Ensure inputs are numpy arrays
    start = np.asarray(start)
    end = np.asarray(end)
    
    if debug:
        print(f"Start vector norm: {np.linalg.norm(start)}")
        print(f"End vector norm: {np.linalg.norm(end)}")
    
    # Normalize the vectors if they aren't already
    start_norm = np.linalg.norm(start)
    end_norm = np.linalg.norm(end)
    
    if start_norm > 0 and end_norm > 0:
        start = start / start_norm
        end = end / end_norm
        if debug:
            print(f"After normalization - Start norm: {np.linalg.norm(start)}, End norm: {np.linalg.norm(end)}")
    
    # Calculate the angle between vectors
    dot_product = np.clip(np.dot(start, end), -1.0, 1.0)
    omega = np.arccos(dot_product)
    
    if debug:
        print(f"Angle between vectors (omega): {omega}")
    
    t = np.linspace(0, 1, steps)
    
    if not use_slerp or omega < 1e-6:
        # Simple linear interpolation
        if debug:
            print("Using linear interpolation")
        interpolated = start[None, :] + t[:, None] * (end - start)[None, :]
    else:
        # Use spherical linear interpolation (slerp)
        if debug:
            print("Using spherical linear interpolation")
        interpolated = (
            np.sin((1 - t) * omega)[:, None] * start[None, :] +
            np.sin(t * omega)[:, None] * end[None, :]
        ) / np.sin(omega)
    
    if debug:
        print(f"First interpolated vector norm: {np.linalg.norm(interpolated[0])}")
        print(f"Last interpolated vector norm: {np.linalg.norm(interpolated[-1])}")
    
    # Only renormalize if using slerp
    if use_slerp:
        norms = np.linalg.norm(interpolated, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        interpolated = interpolated / norms
    
    if debug:
        print(f"After final normalization - First vector norm: {np.linalg.norm(interpolated[0])}")
        print(f"After final normalization - Last vector norm: {np.linalg.norm(interpolated[-1])}")
    
    return interpolated

def create_panorama(image_paths, output_path, image_size=(100, 100)):
    """
    Create a panorama by horizontally concatenating resized images.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save the panorama
        image_size: Tuple of (width, height) for resizing
    """
    if not image_paths:
        print("No images to create panorama")
        return
        
    # Read and resize all images
    resized_images = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                resized_images.append(resized)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    if not resized_images:
        print("No valid images to create panorama")
        return
        
    # Create panorama by horizontally concatenating images
    panorama = np.hstack(resized_images)
    
    # Save the panorama
    cv2.imwrite(str(output_path), panorama)
    print(f"Panorama saved to {output_path}")

async def main():
    # CLONED_COLLECTION_NAME = f"{QDRANT_COLLECTION_NAME}_clone"
    # clone_collection(client, QDRANT_COLLECTION_NAME, CLONED_COLLECTION_NAME)
    # wait_for_collection_ready(client, CLONED_COLLECTION_NAME)

    async_client =  AsyncQdrantClient("localhost")
    sync_client = QdrantClient("localhost")
    CLONED_COLLECTION_NAME = QDRANT_COLLECTION_NAME
    image_paths = get_batch_embeddings.get_image_filepaths_from_folders(
        [IMAGES_TO_MATCH_PATH]
    )
    output={}

    image_embeddings = []

    LERP_STEPS = 500

    for index, image_path in enumerate(image_paths):
        try:
            if image_path != image_paths[index]:
                raise Exception("Image path is not the same as the index")
            if index < len(image_paths)-1:
                imagesource = image_paths[index]
                imagetarget = image_paths[index+1]
                embedding, embedding_flipped = get_image_embedding.get_image_embedding(sync_client, imagesource, CLONED_COLLECTION_NAME)
                    
                target_embedding, target_embedding_flipped = get_image_embedding.get_image_embedding(sync_client, imagetarget, CLONED_COLLECTION_NAME)
                
                image_embeddings.append(
                    ImageEmbedding(
                        embedding,
                        embedding_flipped,
                        lerp(embedding, target_embedding, LERP_STEPS), 
                        imagesource
                        )
                    )
            else:
                image_embeddings.append(
                    ImageEmbedding(
                        None,
                        None,
                        [], 
                        image_path
                        )
                    )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # now for each image we have the source image, and a list of lerp steps.
        # find the closest match to each lerped embedding and create an array of images.
        # then we can do the next image seamlessly
    image_sequence = []
    # Create async handler for this lerp stage
    async_handler = AsyncClosestMatchHandler(
        collection_name=CLONED_COLLECTION_NAME,
        real_client=async_client
    )
            
    for index, lerp_stage in enumerate(image_embeddings):
        try:
            print(f"processing {index} of {len(image_embeddings)}")
            image_sequence.append(ImageWithScore(lerp_stage.filename, 0))
            

            # Create dictionary of embeddings to process
            embeddings_to_process = {
                f"step_{i}": lerp_step for i, lerp_step in enumerate(lerp_stage.lerp_steps)
            }
            
            # Process embeddings asynchronously
            results = await async_handler.process_embeddings(
                embeddings=embeddings_to_process,
                limit=1,
                force_sequential=False
            )
            
            # Process results
            for step_key, result in results.items():
                if result.error:
                    print(f"Error processing {step_key}: {result.error}")
                    continue
                    
                if result.filepaths:
                    image_sequence.append(ImageWithScore(
                        result.filepaths[0],
                        result.scores[0]
                    ))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


    lastimage = None
    unique_images = []  # Store unique image paths for panorama
    for indexer, image in enumerate(image_sequence):
        try:
            if indexer % 100 == 0:
                print(f"copying {indexer} of {len(image_sequence)}")
            if lastimage is not None:
                if image.image_path == lastimage:
                    continue
            shutil.copy2(image.image_path, OUTPUT_PATH / f"{indexer:004d}_{str(image.score).replace('.', '')[:4]}.jpg")
            unique_images.append(image.image_path)  # Add to unique images list
            lastimage = image.image_path
        except Exception as e:
            print(f"Error copying {image.image_path}: {e}")

    # Create panoramas in batches of 30 images
    batch_size = 30
    for i in range(0, len(unique_images), batch_size):
        batch = unique_images[i:i + batch_size]
        batch_number = i // batch_size + 1
        panorama_path = OUTPUT_PATH / f"panorama_batch_{batch_number:03d}.jpg"
        print(f"Creating panorama for batch {batch_number} ({len(batch)} images)")
        create_panorama(batch, panorama_path)






    # 1/0
    # indexer=0
    # for index, (image_path, search_result) in enumerate(output.items()):
    #     # Copy the original image

        
    #     # Insert white squares as visual buffer
    #     white_square = create_white_square(300)
    #     for i in range(5):  # Save 5 white squares
    #         white_square_path = OUTPUT_PATH / f"{indexer:004d}.jpg"
    #         cv2.imwrite(str(white_square_path), white_square)
    #         indexer+=1
    #     shutil.copy2(image_path, OUTPUT_PATH / f"{indexer:004d}.jpg")
    #     indexer+=1
    #     if search_result[0].point.score > 0.95:
    #         red_square = create_red_square(300)
    #         cv2.imwrite(str(OUTPUT_PATH / f"{indexer:004d}.jpg"), red_square)
    #         indexer+=1
    #     # Copy search results
    #     for res in search_result:
    #         if not res.is_flipped:
    #             # load the image and flip it
    #             img = cv2.imread(res.point.payload["filename"])
    #             img = cv2.flip(img, 1)
    #             cv2.imwrite(str(OUTPUT_PATH / f"{indexer:004d}.jpg"), img)
    #             indexer+=1
    #         else:
    #             shutil.copy2(res.point.payload["filename"], OUTPUT_PATH / f"{indexer:004d}.jpg")
    #             indexer+=1


if __name__ == '__main__':
    asyncio.run(main())
    