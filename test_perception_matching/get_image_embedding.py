import generate_embeddings
import qdrant_utils
import get_sequence_images_qdrant
import json
import cv2
import numpy as np

def get_image_embedding(client, image_path: str,collection_name: str, img_in_memory:np.ndarray = None, GRABBED_EMBEDDING_PARAMS: generate_embeddings.ImageEmbeddingParams = None):
    """
    Get embeddings used in collection and apply to image,
    return flipped and non-flipped embeddings
    """
    if GRABBED_EMBEDDING_PARAMS is None:
        vector, random_item, closest_matches, payload = get_sequence_images_qdrant.get_random_item_with_closest_match(
        client,
        collection_name=collection_name,
        limit=1
        )
        # ensure using same embeddings by grabbing it straight from qdrant collection
        GRABBED_EMBEDDING_PARAMS = generate_embeddings.ImageEmbeddingParams(**json.loads(payload["params"]))

    if img_in_memory is None:
        img = cv2.imread(image_path)
    else:
        img = img_in_memory
    if GRABBED_EMBEDDING_PARAMS.mask:
        mask = generate_embeddings.create_circular_mask(img.shape)
    else:
        mask = None
    # Flip the image horizontally (mirror effect)
    img_flipped = cv2.flip(img, 1)
    embedding = generate_embeddings.create_image_embedding(
        img, 
        params=GRABBED_EMBEDDING_PARAMS,
        mask=mask
    )
    embedding_flipped = generate_embeddings.create_image_embedding(
        img_flipped, 
        params=GRABBED_EMBEDDING_PARAMS,
        mask=mask
    )
    return embedding, embedding_flipped


def get_closest_match(client, embedding, embedding_flipped, collection_name: str, limit: int = 5):
    """
    Get the closest match to the embedding
    provide flipped and unflipped emebddings and get back a list of unique images
    provide same embedding if no flipping is required
    not a performant operation, just a proof of concept
    """

    search_result = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    search_result_flipped = client.search(
        collection_name=collection_name,
        query_vector=embedding_flipped,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )

    # Create a list of all results with flipped information
    all_results = [qdrant_utils.ScoredPointWithFlip(pt, is_flipped=True) for pt in search_result_flipped]
    all_results.extend([qdrant_utils.ScoredPointWithFlip(pt, is_flipped=False) for pt in search_result])
    
    # reopulate back
    # Sort by score and take top 10
    sorted_files = sorted(all_results, key=lambda x: x.point.score, reverse=True)
    # remove image if it already exists in the output (to avoid flipped and unflipped versions)
    # do backwards so overwrite the dictionary key of first instance
    unique_images = list({i.point.payload["filename"]:i for i in sorted_files}.values())
    if len(set([i.point.id for i in unique_images])) != len(unique_images):
        raise ValueError("Duplicate images found in the output")
    return unique_images
